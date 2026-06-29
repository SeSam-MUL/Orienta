"""Pre-flight validation for batch indexing jobs."""
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import psutil

from backend.api.services.memory_guardian import MemoryGuardian

logger = logging.getLogger(__name__)


@dataclass
class PreFlightReport:
    checks: List[Dict]  # [{name, status (pass/warn/fail), message}]
    can_start: bool = True
    warnings: List[str] = field(default_factory=list)
    estimated_disk_gb: float = 0.0

    def __post_init__(self):
        for c in self.checks:
            if c["status"] == "fail":
                self.can_start = False
            elif c["status"] == "warn":
                self.warnings.append(c["message"])


class PreFlightChecker:
    """Validates everything before a batch starts."""

    def run(
        self,
        files: List[str],
        phases: List[Dict],
        export_dir: str = "",
        export_formats: List[str] = None,
        auto_export: bool = True,
    ) -> PreFlightReport:
        checks = []
        if export_formats is None:
            export_formats = ["h5_light", "ang", "ctf"]

        # 1. Source files exist
        missing_files = [f for f in files if not os.path.isfile(f)]
        checks.append({
            "name": "source_files_exist",
            "status": "pass" if not missing_files else "fail",
            "message": f"All {len(files)} source files found" if not missing_files
                       else f"Missing: {', '.join(Path(f).name for f in missing_files)}",
        })

        # 2. Phase files exist
        phase_paths = [p["path"] for p in phases]
        missing_phases = [p for p in phase_paths if not os.path.isfile(p)]
        checks.append({
            "name": "phase_files_exist",
            "status": "pass" if not missing_phases else "fail",
            "message": f"All {len(phases)} phase files found" if not missing_phases
                       else f"Missing: {', '.join(Path(p).name for p in missing_phases)}",
        })

        # 3. Disk space — sum budget across all chosen export formats.
        # Previous estimate ("5 MB per phase result") was an order of
        # magnitude low because it ignored that h5_rich copies the full
        # source file. A 20 GB source × 10 files with rich export =
        # ~200 GB — that used to show as "0.5 GB needed".
        total_source_mb = sum(
            os.path.getsize(f) / (1024**2) for f in files if os.path.isfile(f)
        )
        checkpoint_mb_per_file = len(phases) * 5   # _multiphase.h5 + checkpoints
        estimated_result_mb = checkpoint_mb_per_file * len(files)

        if auto_export:
            for fmt in export_formats:
                if fmt == "h5_rich":
                    # Copies source verbatim + small indexing overhead
                    estimated_result_mb += total_source_mb * 1.05
                elif fmt == "h5_light":
                    # Only indexing data, no pattern copy: ~5 MB per phase
                    # plus per-file constant for metadata.
                    estimated_result_mb += (len(phases) * 5 + 5) * len(files)
                elif fmt in ("ang", "ctf"):
                    # Plain-text orientation dumps are pixels * ~80 bytes.
                    # We don't have the grid shape here; estimate 20 MB/file
                    # as a safe upper bound for typical scans.
                    estimated_result_mb += 20 * len(files)
        estimated_disk_gb = estimated_result_mb / 1024

        target_dir = export_dir if export_dir and os.path.isdir(export_dir) else (
            str(Path(files[0]).parent) if files else "."
        )
        try:
            usage = shutil.disk_usage(target_dir)
            free_gb = usage.free / (1024**3)
            # Fail (not warn) when we don't even have 1x the estimate —
            # export will crash mid-write otherwise. Warn at 1.5x.
            if free_gb < estimated_disk_gb:
                status = "fail"
            elif free_gb < estimated_disk_gb * 1.5:
                status = "warn"
            else:
                status = "pass"
            checks.append({
                "name": "disk_space",
                "status": status,
                "message": f"{free_gb:.1f} GB free, ~{estimated_disk_gb:.1f} GB needed "
                           f"(formats: {', '.join(export_formats) if auto_export else 'auto-export off'})",
            })
        except Exception as e:
            checks.append({"name": "disk_space", "status": "warn", "message": str(e)})

        # 4. RAM budget
        mg = MemoryGuardian()
        budget = mg.pre_flight_budget(files, phase_paths)
        checks.append({
            "name": "ram_budget",
            "status": "pass" if budget["fits_in_ram"] else "warn",
            "message": f"Peak ~{budget['peak_estimate_mb']:.0f} MB, available {budget['available_mb']:.0f} MB",
        })

        # 5. Export dir writable
        if export_dir:
            writable = os.path.isdir(export_dir) and os.access(export_dir, os.W_OK)
        else:
            writable = True  # Will use source file directory
            if files:
                parent = str(Path(files[0]).parent)
                writable = os.path.isdir(parent) and os.access(parent, os.W_OK)
        checks.append({
            "name": "export_dir_writable",
            "status": "pass" if writable else "fail",
            "message": "Export directory writable" if writable else f"Cannot write to {export_dir or 'source directory'}",
        })

        # 6. Stale tmp files — cleanup is best-effort. A locked .tmp on Windows
        # (still held by a crashed backend) must not abort the whole preflight
        # with a 500, so the os.remove is tolerant.
        tmp_cleaned = 0
        tmp_locked = 0
        for f in files:
            tmp = str(Path(f).parent / f"{Path(f).stem}_multiphase.h5.tmp")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                    tmp_cleaned += 1
                except OSError as e:
                    logger.warning("Could not remove stale tmp %s: %s", tmp, e)
                    tmp_locked += 1
        if tmp_locked:
            checks.append({
                "name": "stale_tmp_files",
                "status": "warn",
                "message": (f"Cleaned {tmp_cleaned}, but {tmp_locked} .tmp file(s) "
                            f"are locked — close any other running batch first"),
            })
        else:
            checks.append({
                "name": "stale_tmp_files",
                "status": "pass" if not tmp_cleaned else "warn",
                "message": "No stale files" if not tmp_cleaned
                           else f"Cleaned {tmp_cleaned} stale .tmp files",
            })

        # 7. Power source
        try:
            bat = psutil.sensors_battery()
            if bat is not None and not bat.power_plugged:
                checks.append({
                    "name": "power_source",
                    "status": "warn",
                    "message": f"Running on battery ({bat.percent}%). Plug in recommended for batch.",
                })
            else:
                checks.append({"name": "power_source", "status": "pass", "message": "On AC power"})
        except Exception:
            checks.append({"name": "power_source", "status": "pass", "message": "Power check unavailable"})

        return PreFlightReport(checks=checks, estimated_disk_gb=estimated_disk_gb)
