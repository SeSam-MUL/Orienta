"""
LRU Cache Manager - Local file cache with size limit

Manages a local cache directory for downloaded SHT/H5 files.
Tracks file access times and evicts least-recently-used files
when the cache exceeds the configured size limit (default: 20 GB).
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default cache limit: 20 GB
DEFAULT_CACHE_LIMIT_BYTES = 20 * 1024 * 1024 * 1024


@dataclass
class CacheEntry:
    """Metadata for a cached file."""
    path: Path
    size_bytes: int
    last_accessed: float          # Unix timestamp
    source_path: str = ""         # Original server path


class LRUCacheManager:
    """
    Manages a local file cache with LRU eviction.

    Files are tracked by path. When cache size exceeds the limit,
    least-recently-used files are evicted until under limit.

    Cache metadata is persisted to a JSON index file.
    """

    def __init__(self, cache_dir: Path, max_size_bytes: int = DEFAULT_CACHE_LIMIT_BYTES):
        """
        Initialize LRU cache manager.

        Args:
            cache_dir: Directory for cached files
            max_size_bytes: Maximum cache size in bytes (default: 20 GB)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_bytes
        self._entries: Dict[str, CacheEntry] = {}  # key: relative path string

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index
        self._index_path = self.cache_dir / ".cache_index.json"
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if not self._index_path.exists():
            return

        try:
            with open(self._index_path, 'r') as f:
                data = json.load(f)

            for key, entry_data in data.items():
                path = self.cache_dir / key
                if path.exists():
                    self._entries[key] = CacheEntry(
                        path=path,
                        size_bytes=entry_data.get("size_bytes", path.stat().st_size),
                        last_accessed=entry_data.get("last_accessed", time.time()),
                        source_path=entry_data.get("source_path", ""),
                    )

            logger.info(f"Loaded cache index: {len(self._entries)} entries")

        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")

    def _save_index(self):
        """Persist cache index to disk."""
        try:
            data = {}
            for key, entry in self._entries.items():
                data[key] = {
                    "size_bytes": entry.size_bytes,
                    "last_accessed": entry.last_accessed,
                    "source_path": entry.source_path,
                }

            with open(self._index_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get(self, relative_path: str) -> Optional[Path]:
        """
        Get a cached file path, updating access time.

        Args:
            relative_path: Relative path within cache (e.g., "Fe/test.sht")

        Returns:
            Absolute path to cached file, or None if not cached
        """
        entry = self._entries.get(relative_path)
        if entry is None:
            return None

        if not entry.path.exists():
            # File was deleted externally
            del self._entries[relative_path]
            self._save_index()
            return None

        # Update access time
        entry.last_accessed = time.time()
        self._save_index()
        return entry.path

    def put(self, relative_path: str, source_path: str = "") -> Path:
        """
        Register a file in the cache. The file must already exist at the cache path.

        Call this AFTER copying the file to cache_dir / relative_path.

        Args:
            relative_path: Relative path within cache
            source_path: Original server path for reference

        Returns:
            Absolute path to cached file
        """
        full_path = self.cache_dir / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Cache file not found: {full_path}")

        size = full_path.stat().st_size
        self._entries[relative_path] = CacheEntry(
            path=full_path,
            size_bytes=size,
            last_accessed=time.time(),
            source_path=source_path,
        )

        # Evict if over limit
        self._evict_if_needed()
        self._save_index()

        logger.info(f"Cached: {relative_path} ({size / (1024**2):.1f} MB)")
        return full_path

    def contains(self, relative_path: str) -> bool:
        """Check if a file is in the cache."""
        entry = self._entries.get(relative_path)
        if entry is None:
            return False
        if not entry.path.exists():
            del self._entries[relative_path]
            return False
        return True

    def remove(self, relative_path: str) -> bool:
        """
        Remove a file from the cache.

        Args:
            relative_path: Relative path within cache

        Returns:
            True if removed, False if not found
        """
        entry = self._entries.pop(relative_path, None)
        if entry is None:
            return False

        try:
            if entry.path.exists():
                entry.path.unlink()
                logger.info(f"Evicted from cache: {relative_path}")
        except Exception as e:
            logger.error(f"Failed to delete cached file {relative_path}: {e}")

        self._save_index()
        return True

    def total_size(self) -> int:
        """Get total size of cached files in bytes."""
        return sum(e.size_bytes for e in self._entries.values())

    def total_size_gb(self) -> float:
        """Get total size of cached files in GB."""
        return self.total_size() / (1024 ** 3)

    def file_count(self) -> int:
        """Get number of files in cache."""
        return len(self._entries)

    def get_entries(self) -> List[CacheEntry]:
        """Get all cache entries sorted by last accessed (most recent first)."""
        return sorted(self._entries.values(), key=lambda e: e.last_accessed, reverse=True)

    def _evict_if_needed(self):
        """Evict least-recently-used files until under size limit."""
        total = self.total_size()
        if total <= self.max_size_bytes:
            return

        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(self._entries.items(), key=lambda kv: kv[1].last_accessed)

        evicted = 0
        for key, entry in sorted_entries:
            if total <= self.max_size_bytes:
                break

            # Try unlink FIRST. Previous code subtracted from `total`
            # before unlink, then `continue`d on failure — so a locked or
            # already-deleted file made `total` underestimate cache size,
            # which could end the loop early and leave the disk over the
            # budget. Now we only deduct after successful unlink.
            try:
                if entry.path.exists():
                    entry.path.unlink()
            except Exception as e:
                logger.error(f"Failed to evict {key}: {e}")
                continue

            total -= entry.size_bytes
            del self._entries[key]
            evicted += 1
            logger.info(f"LRU evicted: {key} ({entry.size_bytes / (1024**2):.1f} MB)")

        if evicted:
            logger.info(f"Evicted {evicted} files, cache now {total / (1024**3):.2f} GB")

    def clear(self):
        """Clear entire cache."""
        for key in list(self._entries.keys()):
            self.remove(key)
        self._entries.clear()
        self._save_index()
        logger.info("Cache cleared")
