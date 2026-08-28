"""Write a result file so that a crash cannot leave half of one behind.

Measured, not assumed (2026-08-27): a Python process killed while writing an
HDF5 file leaves a file that **opens cleanly**, carries its ``format_version``
attribute, and contains only the datasets flushed so far — indistinguishable
from a finished export. A reviewer of the out-of-memory report made this the
condition for how urgent that bug is: *"vernachlässigbar — vorausgesetzt, der
Absturz hinterlässt keine halbfertige Ergebnisdatei, die später als gültig
durchgeht. Falls doch, rutscht er sofort nach oben."* It did.

The fix is the usual one and it is definitive: build the file under a
neighbouring ``.part`` name and move it onto the final name only after the
writer has finished. ``os.replace`` is atomic within a filesystem, so the
final path only ever holds a complete file — a reader can never observe a
half-written one, no matter when the process dies.

A leftover ``.part`` from an earlier crash is removed when the next export
starts, so the directory does not fill up with debris.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PART_SUFFIX = ".part"


def begin_atomic(final_path: str | Path) -> Path:
    """Start building `final_path`; returns the path to write to instead.

    The functional form, for writers whose body is long enough that wrapping
    it in a `with` would re-indent hundreds of lines and bury the real change
    in a review. Pair with :func:`finish_atomic` at every success return.

    If the writer dies in between, the fragment stays under its `.part` name,
    which nothing reads and the next export deletes.
    """
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + PART_SUFFIX)
    if part.exists():
        logger.warning("Removing leftover partial export: %s", part)
        try:
            part.unlink()
        except OSError as exc:
            logger.warning("Could not remove %s: %s", part, exc)
    return part


def finish_atomic(part_path: str | Path, final_path: str | Path) -> str:
    """Move a finished `.part` onto its final name. Returns the final path."""
    os.replace(Path(part_path), Path(final_path))
    return str(final_path)


class AtomicResultFile:
    """Context manager yielding a temporary path to build the result in.

    On a clean exit the temporary file replaces `final_path`. On an exception
    it is deleted and `final_path` is left untouched — which, for a re-export,
    means the previous good result survives a failed attempt.

        with AtomicResultFile(out) as tmp:
            shutil.copy2(src, tmp)
            with h5py.File(tmp, "a") as f:
                ...
    """

    def __init__(self, final_path: str | Path):
        self.final_path = Path(final_path)
        self.part_path = self.final_path.with_name(self.final_path.name + PART_SUFFIX)

    def __enter__(self) -> str:
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        # Debris from a previous crash — never reuse it, it is half a file.
        if self.part_path.exists():
            logger.warning("Removing leftover partial export: %s", self.part_path)
            try:
                self.part_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove %s: %s", self.part_path, exc)
        return str(self.part_path)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            # Failed attempt: take the fragment with us, keep any previous
            # result intact.
            try:
                if self.part_path.exists():
                    self.part_path.unlink()
            except OSError:
                logger.warning("Could not clean up %s", self.part_path)
            return False
        try:
            os.replace(self.part_path, self.final_path)
        except OSError:
            logger.exception("Could not move %s into place", self.part_path)
            raise
        return False
