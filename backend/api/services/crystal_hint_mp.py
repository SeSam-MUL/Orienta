"""Materials Project client for the Crystal Hint feature.

Lightweight HTTP client (plain `requests`, no `mp-api` dependency) that
queries the Materials Project REST API for candidate phases matching a
sample's chemistry + crystal-system + lattice-parameter filter.

The API key is read from the per-machine user_config_manager — NOT from
env vars and NOT from the project tree. If no key is configured, the
client gracefully returns [] (the Crystal Hint UI will silently skip MP).

API docs: https://docs.materialsproject.org/downloading-data/using-the-api
Endpoint: https://api.materialsproject.org/materials/summary/

Results are cached on disk under `.cache/crystal_hint/mp_<sha256>.json`
with a 7-day TTL, mirroring `crystal_hint_cod.py`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from backend.api.services import user_config_manager as uc

logger = logging.getLogger(__name__)


MP_BASE_URL = "https://api.materialsproject.org/materials/summary/"
CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "crystal_hint"
CACHE_TTL_SECONDS = 7 * 24 * 3600
REQUEST_TIMEOUT_S = 10.0
MAX_RESULTS_CAP = 50    # API rate-limit-friendly default


@dataclass
class MpResult:
    """One match returned from Materials Project."""

    mp_id: str
    formula: str = ""
    space_group: str = ""
    space_group_number: Optional[int] = None
    crystal_system: str = ""
    a_A: Optional[float] = None
    b_A: Optional[float] = None
    c_A: Optional[float] = None
    cif_url: str = ""    # frontend convenience link
    n_sites: Optional[int] = None
    energy_above_hull_eV: Optional[float] = None


def _cache_key(elements, system, a_range_A):
    parts = [
        "|".join(sorted([e.capitalize() for e in elements])),
        system or "any",
        f"{a_range_A[0]:.2f},{a_range_A[1]:.2f}" if a_range_A else "any",
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"mp_{key}.json"


def _load_cache(key: str) -> Optional[list[MpResult]]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        return [MpResult(**r) for r in data.get("results", [])]
    except Exception as exc:
        logger.warning("Failed to load MP cache %s: %s", path, exc)
        return None


def _save_cache(key: str, results: list[MpResult]) -> None:
    path = _cache_path(key)
    try:
        path.write_text(
            json.dumps({"ts": time.time(), "results": [asdict(r) for r in results]}),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save MP cache %s: %s", path, exc)


def _chemsys_subsystems(elements: list[str], max_subsystems: int = 22) -> list[str]:
    """Generate the chemsys strings worth querying for a multi-element sample.

    MP's `chemsys` filter is EXACT match: `chemsys=Al-Mg-Si` only returns
    phases whose elements are exactly {Al, Mg, Si} — Mg2Si (chemsys=Mg-Si)
    is missed. To find precipitate phases relevant to an alloy with N
    elements, we need to iterate over the most-likely subsystems.

    Priority order (high-value first):
      1. The full chemsys (all elements).
      2. ALL binaries (matrix-AND non-matrix). Binaries are where alloy
         IMs live: Al2Cu, Mg2Si, AlFe3, Al7Cu2Fe (matrix-binary), etc.
      3. The matrix element alone (pure-Al / pure-matrix grains).
      4. Other single elements (pure Si, pure Cu, …).
      5. Matrix-containing ternaries (Q-phase, π-phase, …).

    Capped at `max_subsystems` queries to keep total latency bounded.
    For a 7-element sample (e.g. AA226: Al+Si+Cu+Fe+Mg+Zn+Mn):
      1 full + 21 binaries + 7 singles + C(6,2)=15 matrix-ternaries = 44.
    Default cap of 22 covers all binaries + the matrix single + a few
    ternaries. Roughly 22 × 500ms = ~11s if uncached, then instant.
    Returns a deduplicated, ordered list of chemsys strings.
    """
    elements = [e.strip().capitalize() for e in elements if e and e.strip()]
    if not elements:
        return []
    canonical = sorted(set(elements))
    matrix = elements[0]  # assume first listed element is the matrix

    out: list[str] = []

    def _add(elems: list[str]) -> None:
        cs = "-".join(sorted(set(elems)))
        if cs and cs not in out:
            out.append(cs)

    # 1) full chemsys
    _add(canonical)
    # 2) matrix alone (pure-matrix phase = high-priority for indexable grains)
    _add([matrix])
    # 3) matrix-containing binaries (Al2Cu, Al-Si, AlFe-Si, …)
    others = [e for e in canonical if e != matrix]
    for e in others:
        _add([matrix, e])
    # 4) non-matrix binaries (Mg2Si, CuMg, …) — crucial for precipitates
    from itertools import combinations
    for pair in combinations(others, 2):
        _add(list(pair))
    # 5) other singles (pure Si, pure Cu, …)
    for e in others:
        _add([e])
    # 6) matrix-containing ternaries (Q-phase Al-Cu-Mg-Si, …)
    for pair in combinations(others, 2):
        _add([matrix, *pair])

    return out[:max_subsystems]


def _build_query(elements: list[str], system: Optional[str],
                 a_range_A: Optional[tuple[float, float]],
                 max_results: int,
                 chemsys_override: Optional[str] = None) -> dict:
    """Build query parameters for the MP summary endpoint."""
    params: dict = {
        "_fields": (
            "material_id,formula_pretty,symmetry,structure,nsites,"
            "energy_above_hull"
        ),
        "_per_page": min(max_results, MAX_RESULTS_CAP),
    }
    if chemsys_override is not None:
        params["chemsys"] = chemsys_override
    elif elements:
        # Legacy path: single exact-match chemsys with all elements. Most
        # callers should now use search_mp_sync's subsystem iteration —
        # this falls through to "exact-only" semantics here.
        params["chemsys"] = "-".join(sorted(set(e.capitalize() for e in elements)))
    if system:
        # MP uses capitalized system names: Cubic, Hexagonal, Tetragonal, …
        params["crystal_system"] = system.capitalize()
    # Lattice-parameter filter via cell volume bounds: for a cubic phase,
    # V = a³, so a in [lo, hi] → V in [lo³, hi³]. For non-cubic this is a
    # loose pre-filter (post-filtered client-side after we have a, b, c).
    if a_range_A:
        lo, hi = a_range_A
        params["volume_min"] = max(1.0, lo ** 3 * 0.5)   # generous lower bound
        params["volume_max"] = hi ** 3 * 2.0             # and upper
    return params


def _conventional_lattice_a(sg_symbol: str, a_primitive: float) -> float:
    """Convert primitive cubic `a` to conventional cubic `a`.

    MP returns the PRIMITIVE cell by default. For cubic phases:
      - Fm-3m / Fd-3m / F432 / F-43m (F-centred): a_conv = a_prim * √2
      - Im-3m / Im-3 / I-43m / I432 / I-43d (I-centred): a_conv = a_prim * (2)^(1/3)
      - Pm-3m / P-43m / etc (primitive): a_conv = a_prim
    For non-cubic phases or unrecognised symbols, returns a_primitive
    unchanged.
    """
    if not sg_symbol:
        return a_primitive
    s = sg_symbol.strip().lower().replace(" ", "")
    if s.startswith("f"):
        return a_primitive * (2 ** 0.5)
    if s.startswith("i"):
        return a_primitive * (2 ** (1.0 / 3.0))
    return a_primitive


def _parse_mp_doc(doc: dict) -> Optional[MpResult]:
    """Parse one MP API document into MpResult. Returns None on malformed input.

    The MP API returns the PRIMITIVE cell in `structure.lattice` — for
    F-centred cubic phases (Fm-3m, Fd-3m, …) the user-visible `a` value
    in EBSD literature is the CONVENTIONAL one (a_conv = a_prim · √2).
    We convert for cubic structures so the lattice-range filter matches
    what users tabulate (e.g. Si Fd-3m a=5.43, Mg2Si Fm-3m a=6.39).
    """
    mp_id = doc.get("material_id")
    if not mp_id:
        return None
    sym = doc.get("symmetry") or {}
    struct = doc.get("structure") or {}
    lat = struct.get("lattice") or {}

    sg_symbol = str(sym.get("symbol", ""))
    crystal_system = str(sym.get("crystal_system", "")).lower()

    a_prim = float(lat["a"]) if "a" in lat else None
    b_prim = float(lat["b"]) if "b" in lat else None
    c_prim = float(lat["c"]) if "c" in lat else None

    # Cubic phases: convert primitive → conventional. Non-cubic structures
    # keep the MP-reported lattice (which is the standard one for them).
    if crystal_system == "cubic" and a_prim is not None:
        a_conv = _conventional_lattice_a(sg_symbol, a_prim)
        a_A = a_conv
        b_A = a_conv
        c_A = a_conv
    else:
        a_A = a_prim
        b_A = b_prim
        c_A = c_prim

    return MpResult(
        mp_id=str(mp_id),
        formula=doc.get("formula_pretty", "") or "",
        space_group=sg_symbol,
        space_group_number=int(sym["number"]) if sym.get("number") else None,
        crystal_system=crystal_system,
        a_A=a_A, b_A=b_A, c_A=c_A,
        cif_url=f"https://materialsproject.org/materials/{mp_id}",
        n_sites=int(doc["nsites"]) if doc.get("nsites") else None,
        energy_above_hull_eV=(
            float(doc["energy_above_hull"])
            if doc.get("energy_above_hull") is not None else None
        ),
    )


def _single_chemsys_query(
    chemsys: Optional[str],   # None = chemistry-free (no chemsys filter)
    system: Optional[str],
    a_range_A: Optional[tuple[float, float]],
    api_key: str,
    requests_mod,
    per_query_max: int = 20,
    max_retries: int = 2,
) -> list[MpResult]:
    """Run a single MP query for one chemsys. Returns parsed MpResults.

    Retries on 429/5xx with exponential backoff (0.5s, 1.5s) since the
    parallel burst of 22 queries can trip MP's rate limit. On unrecoverable
    failure: returns [] (caller treats as 'this chemsys had no matches').
    """
    import time as _time
    params = _build_query(
        elements=[], system=system, a_range_A=a_range_A,
        max_results=per_query_max, chemsys_override=chemsys,
    )
    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "User-Agent": "Orienta/CrystalHint",
    }
    for attempt in range(max_retries + 1):
        try:
            r = requests_mod.get(
                MP_BASE_URL, params=params, headers=headers,
                timeout=REQUEST_TIMEOUT_S,
            )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                _time.sleep(0.5 * (1 + attempt * 2))
                continue
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as exc:
            if attempt < max_retries:
                logger.debug(
                    "MP query (chemsys=%s) attempt %d failed: %s — retrying",
                    chemsys, attempt + 1, exc,
                )
                _time.sleep(0.5 * (1 + attempt * 2))
                continue
            logger.warning("MP query (chemsys=%s) gave up after %d attempts: %s",
                           chemsys, max_retries + 1, exc)
            return []
    else:
        return []
    docs = payload.get("data") or payload.get("results") or []
    if isinstance(docs, dict):
        docs = [docs]
    out: list[MpResult] = []
    for doc in docs:
        rec = _parse_mp_doc(doc)
        if rec is not None:
            out.append(rec)
    return out


def search_mp_sync(
    elements: list[str],
    system: Optional[str] = None,
    a_range_A: Optional[tuple[float, float]] = None,
    max_results: int = 30,
    use_cache: bool = True,
    max_subsystems: int = 22,
) -> list[MpResult]:
    """Synchronous MP search across the most-relevant subsystems.

    MP's chemsys filter is EXACT-match: chemsys=Al-Mg-Si only returns
    phases whose elements are exactly {Al, Mg, Si} — it misses Mg2Si
    (chemsys=Mg-Si), pure Al, etc. To return everything an alloy user
    cares about we iterate up to `max_subsystems` chemsys queries
    (singles + binaries + matrix-ternaries) and merge results.

    Reads the API key from the user-config (per-machine). If no key is
    configured, returns [] silently — caller treats absence of external
    matches as "MP not configured", not as an error.

    Never raises on network failure; logs + returns [].
    """
    api_key = uc.get_api_key("materials_project")
    if not api_key:
        logger.debug("MP client: no API key configured — skipping query")
        return []

    cache_key = _cache_key(elements, system, a_range_A)
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached[:max_results]

    # Chemistry-free mode (unknown sample): no chemsys iteration — issue a
    # single query constrained by crystal_system / lattice only. Requires
    # at least one of those, else we'd pull the whole database.
    if not elements:
        if system is None and a_range_A is None:
            return []
        try:
            import requests
        except ImportError:
            return []
        results = _single_chemsys_query(
            chemsys=None, system=system, a_range_A=a_range_A,
            api_key=api_key, requests_mod=requests,
            per_query_max=MAX_RESULTS_CAP,
        )
        # MP's volume pre-filter assumes V≈a³ (cubic only); for non-cubic
        # phases it's meaningless, so post-filter on the ACTUAL lattice a
        # (and b/c) when a bound was given. Keep a result if ANY of its
        # axes falls in range — handles elongated cells where the
        # characteristic axis isn't `a`.
        if a_range_A is not None:
            lo, hi = a_range_A
            def _in_range(r):
                return any(v is not None and lo <= v <= hi
                           for v in (r.a_A, r.b_A, r.c_A))
            results = [r for r in results if _in_range(r)]
        results.sort(key=lambda r: (r.energy_above_hull_eV
                                    if r.energy_above_hull_eV is not None
                                    else 9.9))
        results = results[:max_results]
        if use_cache:
            _save_cache(cache_key, results)
        return results

    try:
        import requests
    except ImportError:
        logger.warning("`requests` not available — MP lookup skipped")
        return []

    # Generate subsystems and fire ALL queries in parallel via a thread
    # pool. We can't early-break here — Mg-Si (which contains Mg2Si) lives
    # ~position 18 of 22 in the subsystem list and earlier subsystems
    # might fill the budget with less-relevant phases. So: collect
    # everything, sort, then cap.
    chemsys_list = _chemsys_subsystems(elements, max_subsystems=max_subsystems)
    per_query_max = min(max_results, MAX_RESULTS_CAP)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Group results by their originating chemsys so we can enforce a
    # per-chemsys cap during merging. Without that, chemsys with many
    # stable phases (e.g. Al-Cu) crowd out the only candidate from a
    # high-value chemsys like Mg-Si (Mg2Si).
    per_chemsys: dict[str, list[MpResult]] = {}
    # Keep parallelism modest — MP's rate limit kicks in around 6-8
    # concurrent requests. 4 workers × 22 queries ≈ 6s total.
    max_workers = min(4, len(chemsys_list))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _single_chemsys_query,
                cs, system, a_range_A, api_key, requests, per_query_max,
            ): cs for cs in chemsys_list
        }
        logger.info("MP: submitted %d chemsys queries (chemsys_list=%d)",
                    len(futures), len(chemsys_list))
        for f in as_completed(futures):
            cs = futures[f]
            try:
                new_results = f.result()
            except Exception as exc:
                logger.warning("chemsys query (%s) failed: %s", cs, exc)
                continue
            per_chemsys[cs] = new_results
            logger.info(
                "MP: chemsys=%s → %d results", cs, len(new_results),
            )
    logger.info("MP: per_chemsys total entries: %d", len(per_chemsys))

    # Inside each chemsys: sort by (a_dist, ehull) so the best entry
    # per chemsys gets quoted first.
    def _intra_sort_key(r: MpResult):
        a_dist = 0.0
        if a_range_A and r.a_A is not None:
            lo, hi = a_range_A
            if r.a_A < lo: a_dist = (lo - r.a_A) / max(lo, 1e-3)
            elif r.a_A > hi: a_dist = (r.a_A - hi) / max(hi, 1e-3)
        ehull = r.energy_above_hull_eV if r.energy_above_hull_eV is not None else 1.0
        return (a_dist, ehull)
    for cs in per_chemsys:
        per_chemsys[cs].sort(key=_intra_sort_key)

    # Interleave: take 1 from each chemsys in turn (round-robin) so the
    # final list has phases from every requested subsystem. Stops once
    # max_results is reached or all chemsys are exhausted.
    aggregate: list[MpResult] = []
    seen: set[str] = set()
    PER_CHEMSYS_CAP = max(3, max_results // max(1, max(2, len(chemsys_list) // 2)))
    indices = {cs: 0 for cs in per_chemsys}
    taken_per_cs = {cs: 0 for cs in per_chemsys}
    while len(aggregate) < max_results:
        progress = False
        for cs in chemsys_list:
            if len(aggregate) >= max_results:
                break
            i = indices.get(cs, 0)
            taken = taken_per_cs.get(cs, 0)
            if cs not in per_chemsys or i >= len(per_chemsys[cs]) or taken >= PER_CHEMSYS_CAP:
                continue
            rec = per_chemsys[cs][i]
            indices[cs] = i + 1
            if rec.mp_id in seen:
                continue
            seen.add(rec.mp_id)
            aggregate.append(rec)
            taken_per_cs[cs] = taken + 1
            progress = True
        if not progress:
            break

    # Aggregate is already ordered by the round-robin merge above.
    if use_cache:
        _save_cache(cache_key, aggregate)
    return aggregate[:max_results]


async def search_mp_async(
    elements: list[str],
    system: Optional[str] = None,
    a_range_A: Optional[tuple[float, float]] = None,
    max_results: int = 30,
    use_cache: bool = True,
    max_subsystems: int = 22,
) -> list[MpResult]:
    """Async wrapper — runs the blocking sync search in a thread."""
    import asyncio
    return await asyncio.to_thread(
        search_mp_sync, elements, system, a_range_A,
        max_results, use_cache, max_subsystems,
    )


def test_mp_key(api_key: str) -> tuple[bool, str]:
    """Verify an API key by making a minimal MP request.

    Returns (success, status_message). Never raises.
    Used by the Settings page "Test" button.
    """
    if not api_key or not api_key.strip():
        return False, "Empty key"
    try:
        import requests
    except ImportError:
        return False, "requests not installed"
    try:
        # Minimal request: ask for one well-known stable material
        r = requests.get(
            MP_BASE_URL,
            params={
                "_fields": "material_id,formula_pretty",
                "_per_page": 1,
                "formula": "Al",
            },
            headers={
                "X-API-KEY": api_key,
                "Accept": "application/json",
                "User-Agent": "Orienta/CrystalHint",
            },
            timeout=REQUEST_TIMEOUT_S,
        )
    except Exception as exc:
        return False, f"Network error: {exc}"
    if r.status_code == 200:
        return True, "OK"
    if r.status_code == 401:
        return False, "401 Unauthorized — invalid API key"
    if r.status_code == 403:
        return False, "403 Forbidden — key lacks required permissions"
    if r.status_code == 429:
        return False, "429 Rate limited — try again later"
    return False, f"HTTP {r.status_code}"


def fetch_cif_text(mp_id: str) -> Optional[str]:
    """Fetch CIF text for a Materials Project entry.

    Strategy:
      1. Pull the full structure via the /materials/summary endpoint with
         _fields=structure (already part of the search query, but we ask
         again here in case the user wasn't in a recent search context).
      2. Build the CIF from the pymatgen Structure object — that's what
         the Materials Project front-end downloads too.

    Returns the CIF text on success, None on any failure (no API key,
    network error, unknown mp_id, pymatgen missing, etc.). Caller is
    responsible for downstream parsing + saving.
    """
    api_key = uc.get_api_key("materials_project")
    if not api_key:
        logger.warning("MP cif fetch: no API key configured")
        return None
    try:
        import requests
    except ImportError:
        return None
    url = "https://api.materialsproject.org/materials/summary/"
    params = {
        "_fields": "material_id,structure",
        "material_ids": mp_id,
    }
    try:
        r = requests.get(
            url, params=params,
            headers={"X-API-KEY": api_key, "Accept": "application/json"},
            timeout=10.0,
        )
        r.raise_for_status()
    except Exception as exc:
        logger.warning("MP cif fetch: HTTP failure for %s: %s", mp_id, exc)
        return None

    try:
        data = r.json().get("data", [])
    except ValueError:
        logger.warning("MP cif fetch: invalid JSON for %s", mp_id)
        return None
    if not data:
        logger.warning("MP cif fetch: no result for %s", mp_id)
        return None

    structure_dict = data[0].get("structure")
    if structure_dict is None:
        logger.warning("MP cif fetch: no structure field in response for %s", mp_id)
        return None

    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        logger.warning("MP cif fetch: pymatgen not available")
        return None

    try:
        s = Structure.from_dict(structure_dict)
        # MP returns PRIMITIVE cells. For EBSD SHT generation we need the
        # CONVENTIONAL standard cell (e.g. cubic Al as a=4.05Å Fm-3m,
        # not a=2.86Å P1 rhombohedral). SpacegroupAnalyzer also detects
        # the proper symmetry from the atomic positions and writes
        # symmetry-equivalent positions into the CIF — which is what
        # EMsoft's mkxtal needs to build the master pattern correctly.
        try:
            sga = SpacegroupAnalyzer(s, symprec=0.01)
            s = sga.get_conventional_standard_structure()
            cif_text = str(CifWriter(s, symprec=0.01))
        except Exception as exc:
            # Fall back to primitive if symmetry detection fails; the
            # SHT pipeline can still cope with P1 for triclinic structures.
            logger.warning(
                "MP cif fetch: conventionalisation failed for %s (%s) — "
                "falling back to primitive cell", mp_id, exc,
            )
            cif_text = str(CifWriter(s))
    except Exception as exc:
        logger.warning("MP cif fetch: pymatgen failed to build CIF for %s: %s", mp_id, exc)
        return None
    return cif_text


def _doi_from_bibtex(bibtex: str) -> str:
    """Extract a DOI from a BibTeX entry, '' if none. Handles
    `doi = {10...}` / `doi="10..."` / `DOI = 10...`."""
    import re
    m = re.search(r'doi\s*=\s*[{"\']?\s*(10\.[^\s,}"\']+)', bibtex,
                  re.IGNORECASE)
    return m.group(1).strip().rstrip('.') if m else ""


def _short_citation_from_bibtex(bibtex: str) -> str:
    """Build a one-line 'Author et al., Journal Volume (Year)' citation
    from a BibTeX entry. Returns '' if nothing usable is found."""
    import re

    def _field(name: str) -> str:
        m = re.search(rf'{name}\s*=\s*[{{"\']?(.*?)[}}"\']?\s*,?\s*\n',
                      bibtex, re.IGNORECASE | re.DOTALL)
        return (m.group(1).strip().strip('{}"\' ') if m else "")

    author = _field("author")
    first_author = ""
    if author:
        # "Last, First and Last2, First2" → "Last et al."
        first = author.split(" and ")[0]
        first_author = first.split(",")[0].strip()
        if " and " in author or "," in author:
            first_author += " et al."
    journal = _field("journal")
    year = _field("year")
    volume = _field("volume")
    title = _field("title")
    parts = []
    if first_author:
        parts.append(first_author)
    if title:
        parts.append(f'"{title}"')
    jv = " ".join(p for p in [journal, volume] if p)
    if jv:
        parts.append(jv)
    if year:
        parts.append(f"({year})")
    return ", ".join(parts)


def fetch_mp_reference(mp_id: str) -> dict:
    """Fetch the source reference + DOI for a Materials Project entry.

    Queries MP's provenance endpoint (references = BibTeX list,
    database_IDs = ICSD ids) and returns::
        {"doi": "<first DOI found, or ''>",
         "reference": "<one-line citation + MP/ICSD provenance>"}

    Always returns a dict (never raises). On any failure the reference
    falls back to the Materials Project entry itself, which is a valid
    citation for the structure even when no source paper DOI is exposed.
    """
    mp_id = str(mp_id).strip()
    mp_url = f"https://materialsproject.org/materials/{mp_id}"
    fallback = {
        "doi": "",
        "reference": f"Materials Project {mp_id} ({mp_url}). "
                     f"A. Jain et al., APL Materials 1, 011002 (2013).",
    }
    api_key = uc.get_api_key("materials_project")
    if not api_key:
        return fallback
    try:
        import requests
        r = requests.get(
            "https://api.materialsproject.org/materials/provenance/",
            params={"material_ids": mp_id,
                    "_fields": "material_id,references,database_IDs"},
            headers={"X-API-KEY": api_key, "Accept": "application/json"},
            timeout=12.0,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return fallback
        doc = data[0]
        refs = doc.get("references") or []
        db_ids = doc.get("database_IDs") or {}
        icsd = db_ids.get("icsd") or []

        # Pick the SOURCE structure reference. MP's provenance list always
        # includes the Materials Project's own paper (Jain et al. 2013,
        # doi 10.1063/1.4812323) — skip that so we cite the experimental
        # source, not MP citing itself. Fall back to the MP paper only if
        # no real source reference is exposed.
        MP_SELF_DOI = "10.1063/1.4812323"
        doi = ""
        primary = ""
        for bib in refs:
            d = _doi_from_bibtex(bib)
            if d and d != MP_SELF_DOI:
                doi = d
                primary = _short_citation_from_bibtex(bib)
                break
        if not doi:  # no source DOI — use first non-MP citation text, else first
            for bib in refs:
                if _doi_from_bibtex(bib) != MP_SELF_DOI:
                    primary = _short_citation_from_bibtex(bib)
                    if primary:
                        break
            if not primary and refs:
                primary = _short_citation_from_bibtex(refs[0])

        prov = f"Materials Project {mp_id}"
        if icsd:
            prov += f" (ICSD: {', '.join(str(i) for i in icsd[:5])}" \
                    + (", …" if len(icsd) > 5 else "") + ")"
        reference = (primary + " | " + prov) if primary else (
            prov + ". " + fallback["reference"])
        return {"doi": doi, "reference": reference}
    except Exception as exc:
        logger.warning("MP reference fetch failed for %s: %s", mp_id, exc)
        return fallback


def clear_cache() -> int:
    """Delete all cached MP responses. Returns count deleted."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("mp_*.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count
