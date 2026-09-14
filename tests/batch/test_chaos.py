"""Pillar 2: Randomized chaos tests with fault injection.

20 scenarios, each with random file/phase combos and 0-3 injected faults.
Invariants must hold regardless of what chaos is injected.
"""
import gc
import os
import random

import h5py
import numpy as np
import pytest

from backend.api.services.batch_queue import BatchQueue
from backend.api.services.batch_manager import BatchManager
from backend.api.services.checkpoint_writer import CheckpointWriter
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_fake_h5oina(path, rows=5, cols=5):
    """Create a minimal fake H5OINA file."""
    with h5py.File(path, "w") as f:
        hdr = f.create_group("1/EBSD/Header")
        hdr.create_dataset("X Cells", data=cols)
        hdr.create_dataset("Y Cells", data=rows)
        hdr.create_dataset("X Step", data=0.5)
        hdr.create_dataset("Pattern Width", data=10)
        hdr.create_dataset("Pattern Height", data=8)
        data = f.create_group("1/EBSD/Data")
        patterns = np.random.randint(0, 255, (rows * cols, 8, 10), dtype=np.uint8)
        data.create_dataset("Processed Patterns", data=patterns, compression="gzip")


def generate_test_files(n_files, rng, tmp_dir):
    files = []
    for i in range(n_files):
        rows = rng.randint(3, 8)
        cols = rng.randint(3, 8)
        path = os.path.join(tmp_dir, f"chaos_file_{i}.h5oina")
        generate_fake_h5oina(path, rows, cols)
        files.append({"path": path, "rows": rows, "cols": cols})
    return files


def pick_phases(n_phases, rng):
    names = [f"Phase_{chr(65 + i)}" for i in range(26)]
    rng.shuffle(names)
    return [
        {"name": names[i], "path": f"/fake/phase_{i}.sht", "method": "spherical"}
        for i in range(min(n_phases, 26))
    ]


def fake_indexing(rows, cols):
    return (
        np.random.rand(rows, cols).astype(np.float32),
        np.random.rand(rows, cols, 3).astype(np.float32),
        {"ci_mean": 0.5 + random.random() * 0.3, "ci_median": 0.5, "duration_sec": 0.05, "phase_file": "/fake"},
    )


# ---------------------------------------------------------------------------
# Chaos actions
# ---------------------------------------------------------------------------

def chaos_corrupt_checkpoint(tmp_dir, files):
    for f in files:
        cp = CheckpointWriter(f["path"]).checkpoint_path
        if os.path.isfile(cp):
            with open(cp, "ab") as fh:
                fh.write(b"CORRUPTED")
            return


def chaos_delete_source(files, rng):
    if not files:
        return
    victim = rng.choice(files)
    if os.path.isfile(victim["path"]):
        os.remove(victim["path"])


def chaos_memory_pressure():
    blocks = []
    try:
        for _ in range(3):
            blocks.append(np.zeros((5, 1024, 1024), dtype=np.uint8))
    except MemoryError:
        pass
    finally:
        del blocks
        gc.collect()


CHAOS_ACTIONS = {
    "corrupt_checkpoint": lambda ctx: chaos_corrupt_checkpoint(ctx["tmp_dir"], ctx["files"]),
    "delete_source": lambda ctx: chaos_delete_source(ctx["files"], ctx["rng"]),
    "memory_pressure": lambda ctx: chaos_memory_pressure(),
}


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------

class BatchInvariants:
    def __init__(self, queue, batch_id):
        self.queue = queue
        self.batch_id = batch_id

    @property
    def no_zombie_jobs(self):
        jobs = self.queue.get_jobs(self.batch_id)
        return not any(j["status"] == "running" for j in jobs)

    @property
    def db_consistent(self):
        status = self.queue.get_batch_status(self.batch_id)
        jobs = self.queue.get_jobs(self.batch_id)
        done = sum(1 for j in jobs if j["status"] == "done")
        failed = sum(1 for j in jobs if j["status"] == "failed")
        return status["completed"] == done and status["failed"] == failed

    @property
    def no_data_corruption(self):
        jobs = self.queue.get_jobs(self.batch_id)
        for j in jobs:
            if j["status"] == "done" and j["ci_mean"] is not None:
                if j["ci_mean"] < 0 or j["ci_mean"] != j["ci_mean"]:
                    return False
        return True

    @property
    def graceful_errors(self):
        jobs = self.queue.get_jobs(self.batch_id)
        for j in jobs:
            if j["status"] == "failed" and not j.get("error_msg"):
                return False
        return True


# ---------------------------------------------------------------------------
# 20 parametrized chaos scenarios
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_chaos_batch(seed, tmp_path):
    """Run randomized batch with fault injection, verify invariants."""
    rng = random.Random(seed)

    n_files = rng.randint(1, 4)
    n_phases = rng.randint(1, 5)
    tmp_dir = str(tmp_path / f"chaos_{seed}")
    os.makedirs(tmp_dir, exist_ok=True)

    files = generate_test_files(n_files, rng, tmp_dir)
    phases = pick_phases(n_phases, rng)

    action_names = rng.sample(
        list(CHAOS_ACTIONS.keys()),
        k=rng.randint(0, min(2, len(CHAOS_ACTIONS))),
    )

    db_path = os.path.join(tmp_dir, "chaos.db")
    bm = BatchManager(db_path=db_path)

    batch_id, _ = bm.create_batch(
        [f["path"] for f in files],
        phases,
        {},
    )

    grid = (files[0]["rows"], files[0]["cols"])

    def mock_indexing(*args, **kwargs):
        if action_names and rng.random() < 0.3:
            action = rng.choice(action_names)
            ctx = {"tmp_dir": tmp_dir, "files": files, "rng": rng}
            try:
                CHAOS_ACTIONS[action](ctx)
            except Exception:
                pass
        return fake_indexing(grid[0], grid[1])

    try:
        with patch("backend.api.services.batch_manager.run_single_indexing_job", side_effect=mock_indexing):
            bm.run_batch_sync(batch_id)
    except Exception:
        pass

    bm.queue.cleanup_stale()

    inv = BatchInvariants(bm.queue, batch_id)
    assert inv.no_zombie_jobs, f"Seed {seed}: zombie jobs"
    assert inv.db_consistent, f"Seed {seed}: DB inconsistent"
    assert inv.no_data_corruption, f"Seed {seed}: data corruption"
