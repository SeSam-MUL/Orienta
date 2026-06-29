"""SQLite-based persistent job queue for batch indexing."""
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = str(Path(__file__).parents[3] / "data" / "batch_jobs.db")


class BatchQueue:
    """Persistent job queue backed by SQLite.

    One job = one file x one phase. Jobs are grouped by file for smart
    scheduling (load file once, run all phases, then move to next file).
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id          TEXT PRIMARY KEY,
                    created_at  TEXT,
                    status      TEXT DEFAULT 'pending',
                    config_json TEXT,
                    total_jobs  INTEGER DEFAULT 0,
                    completed   INTEGER DEFAULT 0,
                    failed      INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id     TEXT REFERENCES batches(id) ON DELETE CASCADE,
                    file_path    TEXT,
                    file_name    TEXT,
                    phase_name   TEXT,
                    phase_path   TEXT,
                    method       TEXT,
                    status       TEXT DEFAULT 'pending',
                    ci_mean      REAL,
                    ci_median    REAL,
                    error_msg    TEXT,
                    started_at   TEXT,
                    finished_at  TEXT,
                    duration_sec REAL,
                    retry_count  INTEGER DEFAULT 0,
                    sort_order   INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            """)

    def create_batch(
        self,
        files: List[str],
        phases: List[Dict[str, str]],
        config: Dict[str, Any],
    ) -> str:
        """Create a batch with jobs for every file x phase combination.

        Parameters
        ----------
        files : list of str
            Source H5OINA file paths.
        phases : list of dict
            Each dict has keys: name, path, method.
        config : dict
            Batch-level configuration (auto_export, export_dir, etc.).

        Returns
        -------
        str
            Batch UUID.
        """
        batch_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        total = len(files) * len(phases)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO batches (id, created_at, status, config_json, total_jobs) VALUES (?, ?, 'pending', ?, ?)",
                (batch_id, now, json.dumps(config), total),
            )
            sort_order = 0
            for file_idx, fp in enumerate(files):
                for phase in phases:
                    conn.execute(
                        """INSERT INTO jobs
                        (batch_id, file_path, file_name, phase_name, phase_path, method, sort_order)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            batch_id,
                            fp,
                            Path(fp).stem,
                            phase["name"],
                            phase["path"],
                            phase["method"],
                            sort_order,
                        ),
                    )
                    sort_order += 1
        logger.info("Created batch %s: %d files x %d phases = %d jobs", batch_id, len(files), len(phases), total)
        return batch_id

    def next_job(self, batch_id: str) -> Optional[Dict]:
        """Return the next pending job sorted by file grouping, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE batch_id = ? AND status = 'pending' ORDER BY sort_order LIMIT 1",
                (batch_id,),
            ).fetchone()
            return dict(row) if row else None

    def update_job(
        self,
        job_id: int,
        status: str,
        ci_mean: Optional[float] = None,
        ci_median: Optional[float] = None,
        error_msg: Optional[str] = None,
        duration_sec: Optional[float] = None,
    ):
        """Atomically update a job's status and recalculate batch counters."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            if status == "running":
                conn.execute("UPDATE jobs SET status = ?, started_at = ? WHERE id = ?", (status, now, job_id))
            elif status in ("done", "failed", "skipped"):
                conn.execute(
                    """UPDATE jobs SET status = ?, ci_mean = ?, ci_median = ?,
                       error_msg = ?, finished_at = ?, duration_sec = ? WHERE id = ?""",
                    (status, ci_mean, ci_median, error_msg, now, duration_sec, job_id),
                )
            else:
                conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))

            # Recalculate batch counters. If the job was deleted between the
            # update above and this lookup (concurrent delete_batch), the row
            # is gone — skip the counter refresh rather than TypeError on
            # None['batch_id'] and abort the whole worker.
            job_row = conn.execute("SELECT batch_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job_row is None:
                return
            batch_id = job_row["batch_id"]
            # "completed" for the progress bar = actually finished + resumed
            # from checkpoint. Both states mean "no more work needed here";
            # the JobTable distinguishes them visually via different symbols.
            completed = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE batch_id = ? AND status IN ('done', 'skipped')",
                (batch_id,),
            ).fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM jobs WHERE batch_id = ? AND status = 'failed'", (batch_id,)).fetchone()[0]
            conn.execute("UPDATE batches SET completed = ?, failed = ? WHERE id = ?", (completed, failed, batch_id))

    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Get batch summary. Returns None if batch not found."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_jobs(self, batch_id: str) -> List[Dict]:
        """Get all jobs for a batch, sorted by sort_order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE batch_id = ? ORDER BY sort_order", (batch_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def retry_failed(self, batch_id: str) -> int:
        """Reset all failed jobs to pending (increment retry_count). Returns count."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'pending', error_msg = NULL, retry_count = retry_count + 1 "
                "WHERE batch_id = ? AND status = 'failed'",
                (batch_id,),
            )
            count = cursor.rowcount
            conn.execute("UPDATE batches SET failed = 0 WHERE id = ?", (batch_id,))
            return count

    def set_batch_status(self, batch_id: str, status: str):
        """Set the batch-level status (running, paused, completed, failed)."""
        with self._connect() as conn:
            conn.execute("UPDATE batches SET status = ? WHERE id = ?", (status, batch_id))

    def cleanup_stale(self):
        """On startup: reset any 'running' jobs to 'pending' (implies crash)."""
        with self._connect() as conn:
            count = conn.execute("UPDATE jobs SET status = 'pending' WHERE status = 'running'").rowcount
            if count > 0:
                logger.warning("cleanup_stale: reset %d running jobs to pending", count)
            conn.execute("UPDATE batches SET status = 'paused' WHERE status = 'running'")

    def list_batches(self) -> List[Dict]:
        """List all batches, newest first."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def delete_batch(self, batch_id: str):
        """Delete a batch and all its jobs."""
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE batch_id = ?", (batch_id,))
            conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
