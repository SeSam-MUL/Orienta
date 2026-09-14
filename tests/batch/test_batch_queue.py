"""Tests for SQLite batch job queue."""
import pytest
from backend.api.services.batch_queue import BatchQueue


class TestBatchQueue:

    def test_create_batch_returns_id(self, tmp_db):
        q = BatchQueue(tmp_db)
        batch_id = q.create_batch(
            files=["a.h5", "b.h5"],
            phases=[{"name": "Al", "path": "/p/Al.sht", "method": "spherical"}],
            config={"auto_export": True},
        )
        assert batch_id is not None
        assert len(batch_id) == 36  # UUID format

    def test_create_batch_creates_jobs(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(
            files=["a.h5", "b.h5"],
            phases=[
                {"name": "Al", "path": "/p/Al.sht", "method": "spherical"},
                {"name": "Fe", "path": "/p/Fe.sht", "method": "spherical"},
            ],
            config={},
        )
        status = q.get_batch_status(bid)
        assert status["total_jobs"] == 4  # 2 files x 2 phases
        assert status["completed"] == 0
        assert status["failed"] == 0
        assert status["status"] == "pending"

    def test_jobs_sorted_by_file_grouping(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(
            files=["b.h5", "a.h5"],
            phases=[
                {"name": "Al", "path": "/p/Al.sht", "method": "spherical"},
                {"name": "Fe", "path": "/p/Fe.sht", "method": "spherical"},
            ],
            config={},
        )
        jobs = q.get_jobs(bid)
        # Jobs grouped by file: all phases for file 0 before file 1
        assert jobs[0]["file_path"] == jobs[1]["file_path"]
        assert jobs[2]["file_path"] == jobs[3]["file_path"]
        assert jobs[0]["file_path"] != jobs[2]["file_path"]

    def test_next_job_returns_first_pending(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        job = q.next_job(bid)
        assert job is not None
        assert job["status"] == "pending"
        assert job["file_path"] == "a.h5"

    def test_next_job_returns_none_when_empty(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        job = q.next_job(bid)
        q.update_job(job["id"], status="done", ci_mean=0.5, duration_sec=10.0)
        assert q.next_job(bid) is None

    def test_update_job_sets_status(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        job = q.next_job(bid)
        q.update_job(job["id"], status="done", ci_mean=0.72, duration_sec=15.5)
        status = q.get_batch_status(bid)
        assert status["completed"] == 1

    def test_update_job_failed(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        job = q.next_job(bid)
        q.update_job(job["id"], status="failed", error_msg="OOM")
        status = q.get_batch_status(bid)
        assert status["failed"] == 1

    def test_retry_failed_resets_to_pending(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        job = q.next_job(bid)
        q.update_job(job["id"], status="failed", error_msg="OOM")
        count = q.retry_failed(bid)
        assert count == 1
        job2 = q.next_job(bid)
        assert job2 is not None
        assert job2["retry_count"] == 1

    def test_cleanup_stale_resets_running_jobs(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        job = q.next_job(bid)
        q.update_job(job["id"], status="running")
        # Simulate crash: create new queue instance
        q2 = BatchQueue(tmp_db)
        q2.cleanup_stale()
        job2 = q2.next_job(bid)
        assert job2 is not None
        assert job2["status"] == "pending"

    def test_at_most_one_running_invariant(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(
            files=["a.h5"],
            phases=[
                {"name": "Al", "path": "/p", "method": "spherical"},
                {"name": "Fe", "path": "/p", "method": "spherical"},
            ],
            config={},
        )
        j1 = q.next_job(bid)
        q.update_job(j1["id"], status="running")
        # next_job should skip running jobs
        j2 = q.next_job(bid)
        assert j2 is not None
        assert j2["id"] != j1["id"]

    def test_batch_list(self, tmp_db):
        q = BatchQueue(tmp_db)
        q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        q.create_batch(files=["b.h5"], phases=[{"name": "Fe", "path": "/p", "method": "spherical"}], config={})
        batches = q.list_batches()
        assert len(batches) == 2

    def test_delete_batch(self, tmp_db):
        q = BatchQueue(tmp_db)
        bid = q.create_batch(files=["a.h5"], phases=[{"name": "Al", "path": "/p", "method": "spherical"}], config={})
        q.delete_batch(bid)
        assert q.get_batch_status(bid) is None
