"""Network drive synchronization for shared training data.

Provides ``ServerSync`` to copy HDF5 shard files between a local
``TrainingStore`` directory and a shared network location (e.g. a
mounted network drive or NFS share). The sync is file-level, not
sample-level: whole shard files are copied.

Key design decisions
--------------------
* **Atomic writes** — files are written to a temporary name on the
  destination and renamed on success, so a crash mid-transfer never
  leaves a corrupt shard in the target directory.
* **Conflict resolution** — if the same shard filename exists in both
  local and server, it is skipped (content already present). New shards
  are identified by filename uniqueness (the timestamp+hash naming from
  ``TrainingStore`` ensures uniqueness).
* **Partial-failure resilience** — each file is synced independently.
  If one copy fails (e.g. network drop), the others still succeed. A
  ``SyncResult`` reports what worked and what did not.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Callback signature: (current_file_index, total_files, filename)
ProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Outcome of a synchronization operation.

    Parameters
    ----------
    files_copied : list[str]
        Filenames successfully copied to the destination.
    files_skipped : list[str]
        Filenames already present at the destination (no copy needed).
    files_failed : list[str]
        Filenames that could not be copied (error during transfer).
    errors : dict[str, str]
        Mapping of failed filename to error message.
    elapsed_seconds : float
        Wall-clock time of the sync operation.
    """

    files_copied: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    files_failed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def n_copied(self) -> int:
        """Number of files successfully copied."""
        return len(self.files_copied)

    @property
    def n_skipped(self) -> int:
        """Number of files skipped (already at destination)."""
        return len(self.files_skipped)

    @property
    def n_failed(self) -> int:
        """Number of files that failed to copy."""
        return len(self.files_failed)

    @property
    def success(self) -> bool:
        """True if no files failed."""
        return self.n_failed == 0


# ---------------------------------------------------------------------------
# ServerSync
# ---------------------------------------------------------------------------

class ServerSync:
    """Synchronize HDF5 shard files between local and server directories.

    Parameters
    ----------
    local_dir : str or Path
        Local ``TrainingStore`` directory containing ``shard_*.h5`` files.
    server_dir : str or Path
        Shared network directory for collective training data.

    Raises
    ------
    FileNotFoundError
        If *local_dir* does not exist.

    Notes
    -----
    The server directory is created automatically on first ``push`` if it
    does not exist yet. Files are copied atomically: a temporary file is
    written first, then renamed, so an interrupted transfer never corrupts
    the destination.
    """

    def __init__(
        self,
        local_dir: str | Path,
        server_dir: str | Path,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.server_dir = Path(server_dir)

        if not self.local_dir.is_dir():
            raise FileNotFoundError(
                f"Local directory does not exist: {self.local_dir}. "
                "Create a TrainingStore first."
            )

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _shard_files(directory: Path) -> list[Path]:
        """Return sorted shard HDF5 files in *directory*."""
        if not directory.is_dir():
            return []
        return sorted(directory.glob("shard_*.h5"))

    @staticmethod
    def _copy_atomic(src: Path, dst_dir: Path) -> None:
        """Copy *src* into *dst_dir* atomically via temp file + rename.

        Parameters
        ----------
        src : Path
            Source file to copy.
        dst_dir : Path
            Destination directory. Must exist.

        Raises
        ------
        OSError
            If the copy or rename fails.
        """
        tmp_name = f".{src.name}.tmp"
        tmp_path = dst_dir / tmp_name
        final_path = dst_dir / src.name

        try:
            shutil.copy2(str(src), str(tmp_path))
            tmp_path.rename(final_path)
        except BaseException:
            # Clean up partial temp file on any error
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _sync_direction(
        self,
        source_dir: Path,
        dest_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> SyncResult:
        """Copy new shard files from *source_dir* to *dest_dir*.

        Parameters
        ----------
        source_dir : Path
            Directory to copy FROM.
        dest_dir : Path
            Directory to copy TO (created if needed).
        progress_callback : callable, optional
            Called with ``(current_index, total, filename)`` after each file.

        Returns
        -------
        SyncResult
        """
        t0 = time.monotonic()

        dest_dir.mkdir(parents=True, exist_ok=True)

        source_files = self._shard_files(source_dir)
        dest_names = {p.name for p in self._shard_files(dest_dir)}

        result = SyncResult()

        for i, src_path in enumerate(source_files):
            fname = src_path.name

            if fname in dest_names:
                result.files_skipped.append(fname)
                logger.debug("Skipped (already exists): %s", fname)
            else:
                try:
                    self._copy_atomic(src_path, dest_dir)
                    result.files_copied.append(fname)
                    logger.info("Copied: %s -> %s", fname, dest_dir)
                except OSError as exc:
                    result.files_failed.append(fname)
                    result.errors[fname] = str(exc)
                    logger.warning(
                        "Failed to copy %s: %s", fname, exc
                    )

            if progress_callback is not None:
                progress_callback(i + 1, len(source_files), fname)

        result.elapsed_seconds = time.monotonic() - t0
        return result

    # -- Public API ---------------------------------------------------------

    def push(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> SyncResult:
        """Push local shard files to the server directory.

        Files already present on the server (by filename) are skipped.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with ``(current_index, total, filename)`` per file.

        Returns
        -------
        SyncResult
            Summary of the push operation.
        """
        logger.info(
            "Pushing local -> server: %s -> %s",
            self.local_dir, self.server_dir,
        )
        return self._sync_direction(
            self.local_dir, self.server_dir, progress_callback
        )

    def pull(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> SyncResult:
        """Pull shard files from the server to the local directory.

        Files already present locally (by filename) are skipped.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with ``(current_index, total, filename)`` per file.

        Returns
        -------
        SyncResult
            Summary of the pull operation.
        """
        logger.info(
            "Pulling server -> local: %s -> %s",
            self.server_dir, self.local_dir,
        )
        return self._sync_direction(
            self.server_dir, self.local_dir, progress_callback
        )

    def full_sync(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[SyncResult, SyncResult]:
        """Push then pull — bidirectional synchronization.

        Parameters
        ----------
        progress_callback : callable, optional
            Called during both push and pull phases.

        Returns
        -------
        tuple[SyncResult, SyncResult]
            ``(push_result, pull_result)``.
        """
        push_result = self.push(progress_callback)
        pull_result = self.pull(progress_callback)
        return push_result, pull_result

    def server_stats(self) -> dict[str, int]:
        """Return basic statistics about the server directory.

        Returns
        -------
        dict
            Keys: ``n_shards``, ``total_size_bytes``.
        """
        files = self._shard_files(self.server_dir)
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "n_shards": len(files),
            "total_size_bytes": total_bytes,
        }

    def local_stats(self) -> dict[str, int]:
        """Return basic statistics about the local directory.

        Returns
        -------
        dict
            Keys: ``n_shards``, ``total_size_bytes``.
        """
        files = self._shard_files(self.local_dir)
        total_bytes = sum(f.stat().st_size for f in files)
        return {
            "n_shards": len(files),
            "total_size_bytes": total_bytes,
        }
