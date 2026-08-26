"""Simulation job logs accumulated without any bound (81 files, no cleanup)."""

import time

import pytest

from backend.api.routes import simulation


@pytest.fixture()
def log_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(simulation, "_SIM_LOG_DIR", tmp_path)
    return tmp_path


def _make_logs(directory, count, start_mtime=1_000_000):
    paths = []
    for i in range(count):
        p = directory / f"job-{i:03d}.log"
        p.write_text(f"log {i}\n", encoding="utf-8")
        # Deterministic ordering: higher index = newer
        import os
        os.utime(p, (start_mtime + i, start_mtime + i))
        paths.append(p)
    return paths


def test_prune_keeps_the_newest(log_dir):
    _make_logs(log_dir, 60)
    removed = simulation._prune_job_logs(keep=50)
    assert removed == 10
    remaining = sorted(p.name for p in log_dir.glob("*.log"))
    assert len(remaining) == 50
    # The ten oldest went, the newest stayed
    assert "job-000.log" not in remaining
    assert "job-009.log" not in remaining
    assert "job-010.log" in remaining
    assert "job-059.log" in remaining


def test_prune_is_a_noop_below_the_limit(log_dir):
    _make_logs(log_dir, 5)
    assert simulation._prune_job_logs(keep=50) == 0
    assert len(list(log_dir.glob("*.log"))) == 5


def test_prune_tolerates_a_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(simulation, "_SIM_LOG_DIR", tmp_path / "does-not-exist")
    assert simulation._prune_job_logs() == 0  # no raise


def test_writing_a_job_log_prunes(log_dir, monkeypatch):
    monkeypatch.setattr(simulation, "_MAX_SIM_LOGS", 3)
    _make_logs(log_dir, 5)
    simulation._write_job_log("new-task", ["line one"], crystal="Al")
    files = list(log_dir.glob("*.log"))
    assert len(files) == 3
    names = {p.name for p in files}
    assert "new-task.log" in names  # the one just written survives
    assert (log_dir / "new-task.log").read_text(encoding="utf-8").count("line one") == 1
