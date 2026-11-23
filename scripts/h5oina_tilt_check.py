"""Check .h5oina files for tilt distortion from the command line.

Runs the same check the app runs when it loads a file
(``backend/api/services/tilt_distortion.py``) and prints one JSON object per
file followed by a one-line summary.

    python scripts/h5oina_tilt_check.py "Messungen/*.h5oina"
    python scripts/h5oina_tilt_check.py file1.h5oina file2.h5oina --summary

Use it to check the reference cases in docs/user-guide/TiltDistortion.md on
your own data: the measured ``lx_ly_median`` of each SE image is printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.services.tilt_distortion import check_h5oina  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help=".h5oina files or glob patterns")
    ap.add_argument("--summary", action="store_true",
                    help="only the one-line summary per file (with Lx/Ly of the SE images)")
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.files:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])

    status = 0
    for p in paths:
        try:
            r = check_h5oina(p)
        except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the rest
            print(f"{p}: Fehler: {exc}", file=sys.stderr)
            status = 1
            continue
        if args.summary:
            ratios = ", ".join(
                f"{b['name']}: {b['lx_ly_median']}" for b in r["bilder"]
                if b.get("kanal") == "SE" and b.get("lx_ly_median") is not None)
            print(f"{r['datei']}: Kippung {r['kippung_header_grad']}° | Lx/Ly {ratios or '–'} | "
                  f"{r['zusammenfassung']}")
        else:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print(r["zusammenfassung"])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
