"""Import EBSD scan files into a TrainingStore.

Usage
-----
.. code-block:: bash

    .venv/bin/python scripts/import_data.py scan1.ang --store ./training_data
    .venv/bin/python scripts/import_data.py *.ctf --store ./data --ci 0.4
    .venv/bin/python scripts/import_data.py scan.h5oina --store ./data --dry-run

The script:
1. Detects the file format (ANG, CTF, H5OINA, kikuchipy, EMsoft)
2. Parses each file into a unified ScanData representation
3. Filters points by confidence index (CI) threshold
4. Stores high-confidence samples in the TrainingStore (HDF5)
5. Prints a summary of imported data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Import EBSD scan files into a TrainingStore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s scan.ang --store ./training_data\n"
            "  %(prog)s *.ctf --store ./data --ci 0.4\n"
            "  %(prog)s scan.h5oina --store ./data --dry-run\n"
        ),
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="EBSD scan file(s) to import (.ang, .ctf, .h5, .hdf5, .h5oina)",
    )
    parser.add_argument(
        "--store",
        required=True,
        help="Path to the TrainingStore directory",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.3,
        metavar="THRESHOLD",
        help="Minimum confidence index for import (default: 0.3)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        metavar="SIZE",
        help="Normalized pattern size in pixels (default: 128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and show stats without writing to the store",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output per file",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the import pipeline.

    Parameters
    ----------
    argv : list[str], optional
        Argument list for testing.

    Returns
    -------
    int
        Exit code (0 = success, 1 = some files failed).
    """
    args = parse_args(argv)

    from ebsd_ai.data.scan_import import (
        ImportResult,
        ScanData,
        import_to_store,
        parse_scan,
    )
    from ebsd_ai.data.training_store import TrainingStore

    store_path = Path(args.store)
    files = [Path(f) for f in args.files]

    # Validate files exist
    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: File not found: {f}", file=sys.stderr)
        return 1

    print(f"Import target: {store_path}")
    print(f"CI threshold:  {args.ci}")
    print(f"Pattern size:  {args.target_size}x{args.target_size}")
    print(f"Files:         {len(files)}")
    if args.dry_run:
        print("Mode:          DRY RUN (no data written)")
    print()

    # Create store (unless dry run)
    store: TrainingStore | None = None
    if not args.dry_run:
        store = TrainingStore(local_path=store_path)

    results: list[ImportResult] = []
    errors: list[tuple[Path, str]] = []

    for filepath in files:
        print(f"  {filepath.name} ... ", end="", flush=True)

        try:
            if args.dry_run:
                # Parse only, don't import
                scan = parse_scan(filepath)
                result = _dry_run_result(scan, filepath, args.ci)
            else:
                assert store is not None
                result = import_to_store(
                    filepath,
                    store,
                    ci_threshold=args.ci,
                    target_size=args.target_size,
                )

            results.append(result)

            if result.n_points_imported > 0 or args.dry_run:
                print(
                    f"{result.n_points_imported}/{result.n_points_total} "
                    f"points ({result.format.value})"
                )
            elif result.n_points_skipped_no_pattern > 0:
                print("SKIPPED (no patterns)")
            else:
                print(
                    f"0/{result.n_points_total} points "
                    f"(none above CI >= {args.ci})"
                )

            if args.verbose and result.warnings:
                for w in result.warnings:
                    print(f"    WARNING: {w}")
            if args.verbose and result.phase_counts:
                for phase, count in sorted(result.phase_counts.items()):
                    print(f"    {phase}: {count}")

        except Exception as exc:
            print(f"ERROR: {exc}")
            errors.append((filepath, str(exc)))

    # Summary
    print()
    _print_summary(results, errors, args.dry_run)

    # Show store stats
    if store is not None and not args.dry_run:
        stats = store.get_dataset_stats()
        print(f"\nStore stats: {stats}")

    return 1 if errors else 0


def _dry_run_result(
    scan: "ScanData",
    filepath: Path,
    ci_threshold: float,
) -> "ImportResult":
    """Build an ImportResult without writing to a store."""
    import numpy as np

    from ebsd_ai.data.scan_import import ImportResult

    result = ImportResult(
        source_file=str(filepath),
        format=scan.source_format,
        n_points_total=scan.n_points,
        ci_threshold=ci_threshold,
        has_eds=scan.has_eds,
    )

    issues = scan.validate()
    if issues:
        result.warnings.extend(issues)

    if not scan.has_patterns:
        result.n_points_skipped_no_pattern = scan.n_points
        result.warnings.append("No patterns in file")
        return result

    # Count how many would pass CI threshold
    mask = scan.confidence_scores >= ci_threshold
    result.n_points_imported = int(np.sum(mask))

    if mask.any():
        for idx in np.where(mask)[0]:
            pid = int(scan.phase_ids[idx])
            name = scan.phase_names[pid]
            result.phase_counts[name] = result.phase_counts.get(name, 0) + 1

    return result


def _print_summary(
    results: list["ImportResult"],
    errors: list[tuple[Path, str]],
    dry_run: bool,
) -> None:
    """Print a summary of all import operations."""
    total_points = sum(r.n_points_total for r in results)
    total_imported = sum(r.n_points_imported for r in results)
    total_skipped_np = sum(r.n_points_skipped_no_pattern for r in results)

    action = "Would import" if dry_run else "Imported"

    print(f"--- Summary ---")
    print(f"Files processed: {len(results)}")
    print(f"Files with errors: {len(errors)}")
    print(f"Total scan points: {total_points}")
    print(f"{action}: {total_imported}")
    if total_skipped_np > 0:
        print(f"Skipped (no patterns): {total_skipped_np}")

    # Aggregate phase counts
    all_phases: dict[str, int] = {}
    for r in results:
        for phase, count in r.phase_counts.items():
            all_phases[phase] = all_phases.get(phase, 0) + count

    if all_phases:
        print(f"\nPhase breakdown:")
        for phase, count in sorted(
            all_phases.items(), key=lambda x: -x[1]
        ):
            print(f"  {phase}: {count}")

    if errors:
        print(f"\nErrors:")
        for filepath, msg in errors:
            print(f"  {filepath}: {msg}")


if __name__ == "__main__":
    sys.exit(main())
