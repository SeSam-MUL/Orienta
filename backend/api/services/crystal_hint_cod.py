"""COD (Crystallography Open Database) client for the Crystal Hint feature.

COD is open and free; no API key required. Queries are HTTP GET with
URL-encoded element/spacegroup/cell-parameter filters. The response is a
TSV-ish table from result.php.

Public docs: https://www.crystallography.net/cod/result
Example query URL:
  https://www.crystallography.net/cod/result?
    el1=Al&el2=Si&strictmin=2&strictmax=3&
    a=3.5,7.0&format=tsv

The result columns vary; we parse defensively and return a list of CodResult.

Results are cached on disk under .cache/crystal_hint/cod_<sha256>.json with
a 7-day TTL.

See docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md Section 4.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


COD_BASE_URL = "https://www.crystallography.net/cod/result"
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "crystal_hint"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days
REQUEST_TIMEOUT_S = 8.0


@dataclass
class CodResult:
    """One match returned from COD."""

    cod_id: str
    formula: str = ""
    space_group: str = ""
    space_group_number: Optional[int] = None
    a_A: Optional[float] = None
    b_A: Optional[float] = None
    c_A: Optional[float] = None
    alpha_deg: Optional[float] = None
    beta_deg: Optional[float] = None
    gamma_deg: Optional[float] = None
    cif_url: str = ""
    title: str = ""
    raw_row: dict = field(default_factory=dict)


def _cache_key(elements: list[str], system: Optional[str],
               a_range_A: Optional[tuple[float, float]]) -> str:
    parts = [
        "|".join(sorted([e.capitalize() for e in elements])),
        system or "any",
        f"{a_range_A[0]:.2f},{a_range_A[1]:.2f}" if a_range_A else "any",
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"cod_{key}.json"


def _load_cache(key: str) -> Optional[list[CodResult]]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("ts", 0)
        if time.time() - ts > CACHE_TTL_SECONDS:
            return None
        return [CodResult(**r) for r in data.get("results", [])]
    except Exception as exc:
        logger.warning("Failed to load COD cache %s: %s", path, exc)
        return None


def _save_cache(key: str, results: list[CodResult]) -> None:
    path = _cache_path(key)
    try:
        path.write_text(
            json.dumps({"ts": time.time(), "results": [asdict(r) for r in results]}),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save COD cache %s: %s", path, exc)


def _build_query(
    elements: list[str],
    a_range_A: Optional[tuple[float, float]] = None,
) -> dict[str, str]:
    """Build URL parameters for the COD result.php endpoint."""
    params: dict[str, str] = {"format": "tsv"}
    # COD restricts element filters to el1..el8 slots
    for i, el in enumerate(elements[:8], 1):
        params[f"el{i}"] = el.capitalize()
    # Strict element-count match — element count must be within [min, max]
    # We don't constrain count tightly; default is no constraint.
    if a_range_A is not None:
        # COD wants a=lo,hi (Angstroms)
        params["a"] = f"{a_range_A[0]:.3f},{a_range_A[1]:.3f}"
    return params


def _parse_tsv_row(headers: list[str], cells: list[str]) -> dict[str, str]:
    """Best-effort row parsing — COD's column set varies by query."""
    return {h: cells[i] if i < len(cells) else "" for i, h in enumerate(headers)}


def _matches_system(result: "CodResult", system: str) -> bool:
    """Map a COD result to a crystal system. Prefer space-group NUMBER
    when present (unambiguous); fall back to H-M name parsing.

    System boundaries (International Tables):
      triclinic 1-2, monoclinic 3-15, orthorhombic 16-74,
      tetragonal 75-142, trigonal 143-167, hexagonal 168-194, cubic 195-230.
    """
    if result.space_group_number is not None:
        n = result.space_group_number
        if system == "cubic":
            return 195 <= n <= 230
        if system == "hexagonal":
            return 168 <= n <= 194
        if system == "trigonal":
            return 143 <= n <= 167
        if system == "tetragonal":
            return 75 <= n <= 142
        if system == "orthorhombic":
            return 16 <= n <= 74
        if system == "monoclinic":
            return 3 <= n <= 15
        if system == "triclinic":
            return 1 <= n <= 2
        return True
    # Fallback: heuristic on H-M name (only used when SG number missing)
    if not result.space_group:
        return True
    sg = result.space_group.lower().replace(" ", "")
    cubic_starts = ("fm-3", "im-3", "pm-3", "fd-3", "ia-3", "f-43", "p-43", "f432", "p432", "i432")
    hex_starts = ("p6", "p-6")
    trig_starts = ("p3", "p-3", "r3", "r-3")
    if system == "cubic":
        return any(sg.startswith(t) for t in cubic_starts)
    if system == "hexagonal":
        return any(sg.startswith(t) for t in hex_starts)
    if system == "trigonal":
        return any(sg.startswith(t) for t in trig_starts)
    if system == "tetragonal":
        return any(sg.startswith(t) for t in ("p4", "i4", "p-4", "i-4"))
    if system == "orthorhombic":
        return any(sg.startswith(t) for t in ("p2_12", "p2_22", "p222", "i222", "f222", "pmmm", "pnnn", "pccm", "pmma", "pnma", "pbca", "cmcm", "cccm", "ibam", "imm", "fdd", "fmm", "ccc", "pbcm", "pmmn", "p2_1/c", "pna2"))
    if system == "monoclinic":
        return any(sg.startswith(t) for t in ("p2", "c2", "p2_1", "p21", "p2/c", "p2_1/c", "p2/m", "p2_1/m", "c2/c", "c2/m"))
    return True


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s) if s and s.strip() != "?" else None
    except (TypeError, ValueError):
        return None


def _row_to_result(row: dict[str, str]) -> Optional[CodResult]:
    cod_id = row.get("file") or row.get("id") or row.get("cod_id")
    if not cod_id:
        return None
    return CodResult(
        cod_id=str(cod_id).strip(),
        formula=row.get("formula") or row.get("chemname") or "",
        space_group=row.get("sg") or row.get("spacegroup") or "",
        space_group_number=int(row["sgNumber"]) if row.get("sgNumber", "").strip().isdigit() else None,
        a_A=_safe_float(row.get("a", "")),
        b_A=_safe_float(row.get("b", "")),
        c_A=_safe_float(row.get("c", "")),
        alpha_deg=_safe_float(row.get("alpha", "")),
        beta_deg=_safe_float(row.get("beta", "")),
        gamma_deg=_safe_float(row.get("gamma", "")),
        cif_url=f"https://www.crystallography.net/cod/{cod_id}.cif",
        title=row.get("title") or row.get("commonname") or "",
        raw_row=row,
    )


def search_cod_sync(
    elements: list[str],
    system: Optional[str] = None,
    a_range_A: Optional[tuple[float, float]] = None,
    max_results: int = 30,
    use_cache: bool = True,
) -> list[CodResult]:
    """Synchronous COD search. Returns ranked list of CodResult.

    NOTE: this is a blocking HTTP call. For the FastAPI hot path use
    `search_cod_async` and wait via asyncio.

    If COD is unreachable / offline, returns []. We never raise on network
    failure — caller decides whether the absence of external matches is fatal.
    """
    cache_key = _cache_key(elements, system, a_range_A)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached[:max_results]

    # Chemistry-free mode (unknown sample): COD can only constrain by
    # lattice ('a' range) in the query — crystal system is filtered
    # client-side. So a chemistry-free COD query is only safe with a
    # lattice bound; system-only would be an unbounded full-DB query, so
    # we defer that to MP (which supports a crystal_system filter) and
    # return [] here rather than hammering COD.
    if not elements and a_range_A is None:
        return []

    params = _build_query(elements, a_range_A=a_range_A)
    url = f"{COD_BASE_URL}?{urlencode(params)}"

    try:
        import requests
    except ImportError:
        logger.warning("`requests` not available — COD lookup skipped")
        return []

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT_S, headers={
            "User-Agent": "Orienta/CrystalHint (crystal-search)",
        })
        r.raise_for_status()
    except Exception as exc:
        logger.warning("COD query failed (%s): %s", url, exc)
        return []

    # COD's response is tab-separated (with a leading header line)
    raw_text = r.text
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return []
    # Header: first line, columns separated by tabs
    header = [h.strip() for h in lines[0].split("\t")]
    results: list[CodResult] = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("\t")]
        row = _parse_tsv_row(header, cells)
        result = _row_to_result(row)
        if result is None:
            continue
        # Optional system filter (post-filter — COD doesn't accept system param directly)
        if system is not None:
            if not _matches_system(result, system):
                continue
        results.append(result)
        if len(results) >= max_results:
            break

    if use_cache:
        _save_cache(cache_key, results)
    return results


async def search_cod_async(
    elements: list[str],
    system: Optional[str] = None,
    a_range_A: Optional[tuple[float, float]] = None,
    max_results: int = 30,
    use_cache: bool = True,
) -> list[CodResult]:
    """Async wrapper — runs the blocking sync search in a thread."""
    import asyncio
    return await asyncio.to_thread(
        search_cod_sync, elements, system, a_range_A, max_results, use_cache
    )


def clear_cache() -> int:
    """Delete all cached COD responses. Returns count deleted."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("cod_*.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count
