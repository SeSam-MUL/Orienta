"""Memory Guardian — RAM monitoring and OOM prevention for batch processing."""
import enum
import gc
import logging
import os
from typing import Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

# Decompression factor: H5 files are often gzip-compressed.
# On-disk size x this factor ~ in-memory size.
_DECOMPRESSION_FACTOR = 3.0
_OVERHEAD_MB = 500  # Base overhead for Python + libs + FastAPI


class MemoryStatus(enum.Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class MemoryGuardian:
    """Monitors system RAM and prevents OOM crashes during batch processing."""

    def __init__(self, min_free_gb: float = 2.0, max_usage_pct: float = 80.0):
        self.min_free_gb = min_free_gb
        self.max_usage_pct = max_usage_pct

    def get_current_usage(self) -> Dict:
        """Current RAM statistics for the dashboard."""
        vm = psutil.virtual_memory()
        proc = psutil.Process(os.getpid())
        return {
            "total_mb": vm.total / (1024 ** 2),
            "available_mb": vm.available / (1024 ** 2),
            "percent": vm.percent,
            "process_rss_mb": proc.memory_info().rss / (1024 ** 2),
        }

    def check_can_proceed(self) -> MemoryStatus:
        """Check if there is enough free RAM to start the next job."""
        vm = psutil.virtual_memory()
        free_gb = vm.available / (1024 ** 3)
        if free_gb < self.min_free_gb or vm.percent > 90:
            return MemoryStatus.CRITICAL
        if vm.percent > self.max_usage_pct:
            return MemoryStatus.WARNING
        return MemoryStatus.OK

    def estimate_job_memory(self, file_path: str, dict_path: Optional[str] = None) -> float:
        """Estimate peak RAM in MB for indexing one file against one phase.

        Uses file size on disk x decompression factor + overhead.
        """
        file_mb = os.path.getsize(file_path) / (1024 ** 2) * _DECOMPRESSION_FACTOR
        dict_mb = 0.0
        if dict_path and os.path.isfile(dict_path):
            dict_mb = os.path.getsize(dict_path) / (1024 ** 2) * _DECOMPRESSION_FACTOR
        return file_mb + dict_mb + _OVERHEAD_MB

    def pre_flight_budget(self, files: List[str], dict_paths: List[str]) -> Dict:
        """Full RAM budget analysis before batch start.

        Returns a report dict with fits_in_ram, largest_file_mb, etc.
        """
        vm = psutil.virtual_memory()
        available_mb = vm.available / (1024 ** 2)

        file_sizes = [os.path.getsize(f) / (1024 ** 2) for f in files if os.path.isfile(f)]
        dict_sizes = [os.path.getsize(d) / (1024 ** 2) for d in dict_paths if os.path.isfile(d)]

        largest_file_mb = max(file_sizes) if file_sizes else 0
        largest_dict_mb = max(dict_sizes) if dict_sizes else 0

        peak_estimate_mb = (largest_file_mb * _DECOMPRESSION_FACTOR
                            + largest_dict_mb * _DECOMPRESSION_FACTOR
                            + _OVERHEAD_MB)

        return {
            "available_mb": available_mb,
            "largest_file_mb": largest_file_mb,
            "largest_dict_mb": largest_dict_mb,
            "peak_estimate_mb": peak_estimate_mb,
            "fits_in_ram": peak_estimate_mb < available_mb * 0.8,
            "total_ram_mb": vm.total / (1024 ** 2),
        }

    def recommend_n_per_iteration(self, available_mb: float, n_patterns: int) -> Optional[int]:
        """Calculate safe n_per_iteration for adaptive chunking.

        Returns None if no chunking needed (enough RAM).
        """
        # Each pattern ~20KB in memory (128x156 uint8)
        pattern_mb = n_patterns * 20 / 1024
        if pattern_mb < available_mb * 0.5:
            return None  # No chunking needed
        # Chunk so each iteration uses ~40% of available RAM
        safe_patterns = int(available_mb * 0.4 * 1024 / 20)
        return max(100, safe_patterns)  # At least 100 patterns per iteration

    def force_cleanup(self):
        """Aggressive memory reclamation after a job completes."""
        gc.collect()
        gc.collect()  # Second pass catches reference cycles
        logger.debug("force_cleanup: gc.collect() done, RSS=%.0f MB",
                      psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))

    def is_on_battery(self) -> bool:
        """Check if the system is running on battery power."""
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return False  # Desktop PC, no battery
            return not bat.power_plugged
        except Exception:
            return False
