"""Crystal Hint route — detect crystal symmetry + suggest candidate phases.

Endpoints:
  POST /api/crystal-hint/analyze-pixel       Single-pixel deep analysis (Mode 1)
  POST /api/crystal-hint/external-search     Trigger COD search for missing phases
  GET  /api/crystal-hint/presets             List material presets
  GET  /api/crystal-hint/presets/{key}       Get one preset
  GET  /api/crystal-hint/library             List local CIF/SHT library entries

Spec: docs/superpowers/specs/2026-05-26-crystal-hint-feature-design.md
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.services import h5_session
from backend.api.services.crystal_hint_cif_downloader import (
    CifDownloadResult,
    download_cif_from_cod,
    download_cif_from_mp,
)
from backend.api.services.crystal_hint_cod import CodResult, search_cod_async
from backend.api.services.crystal_hint_mp import MpResult, search_mp_async
from backend.api.services.image_utils import array_to_base64_png, array_to_base64_raw
from backend.api.services.crystal_hint_lattice import LatticeReport, estimate_lattice
from backend.api.services.crystal_hint_local_library import (
    LocalMatch,
    get_index,
    search as library_search,
)
from backend.api.services.crystal_hint_region import (
    RegionAnalysisResult,
    analyze_region,
)
from backend.api.services.crystal_hint_quality import (
    QualityCheckResult,
    quality_check_region,
)
from backend.api.services.crystal_hint_presets import (
    boost_score_for_expected_phase,
    get_preset,
    get_presets,
)
from backend.api.services.crystal_hint_symmetry import (
    SymmetryReport,
    detect_symmetry,
)
from backend.api.services.crystal_hint_spherical import (
    detect_symmetry_spherical,
    detect_symmetry_spherical_averaged,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalyzePixelRequest(BaseModel):
    pixel_index: int = Field(..., ge=0, description="Flat pattern index in EBSD signal")
    elements: list[str] = Field(default_factory=list,
                                description="Sample chemistry (element symbols)")
    preset_key: Optional[str] = Field(None,
                                       description="Material preset for ranking boost")
    crystal_system_filter: Optional[str] = Field(None,
                                                  description="Only return matches in this system")
    strict_chemistry: bool = Field(True,
                                    description="Drop phases with elements outside sample chemistry")
    # Restricted to the two values the H5OINADataExtractor actually
    # supports — 'raw' returns 'Unprocessed Patterns' on Oxford files,
    # everything else maps to 'Processed Patterns'. Used to silently accept
    # bogus values like 'background'.
    pattern_type: Literal["raw", "processed"] = Field("processed",
                              description="Which pattern variant.")
    # Neighborhood averaging — recovers ~2x NCC by suppressing per-pixel
    # noise. Off (radius=0 → single pixel) preserves the precise location;
    # 1 → 3x3 = 9 patterns; 2 → 5x5 = 25 patterns. Most useful at radius=1.
    avg_radius: int = Field(0, ge=0, le=3,
                            description="0 = single pixel; 1 = 3x3 average; 2 = 5x5 average")
    # EDS chemistry weighting (Option C). "off" = pattern-only ranking
    # (bit-identical to before); "soft" = damp score by chemistry match
    # (never to zero); "filter" = drop chemically inconsistent phases.
    # Silently degrades to "off" when the active dataset carries no EDS.
    eds_weighting: Literal["off", "soft", "filter"] = Field("soft",
        description="How per-pixel EDS chemistry weights the ranking.")
    eds_filter_threshold: float = Field(0.35, ge=0.0, le=1.0,
        description="In 'filter' mode, drop phases with chemistry_fit below this.")


class CandidateOut(BaseModel):
    """Single candidate match in the response."""

    key: str
    formula: str
    # Frontend-friendly form ("α-Al(Fe,Mn)Si" instead of raw decimal-heavy
    # CIF formula). Empty if the raw formula was already readable.
    display_formula: str = ""
    space_group: str = ""
    space_group_number: Optional[int] = None
    crystal_system: str = "unknown"
    elements: list[str] = []
    lattice_a_A: Optional[float] = None
    lattice_b_A: Optional[float] = None
    lattice_c_A: Optional[float] = None
    n_atoms: Optional[int] = None
    has_sht: bool = False
    plausibility: float = 0.0
    lattice_distance: float = 0.0
    system_match: bool = True
    score: float = 0.0
    expected_in_preset: bool = False
    # Per-phase fit scores (0..1) — null when not computed
    symmetry_fit: Optional[float] = None
    dspacing_fit: Optional[float] = None
    # EDS chemistry match (0..1) of the pixel's measured At% vs this phase's
    # nominal composition. Null when EDS weighting is off/unavailable.
    chemistry_fit: Optional[float] = None
    # Display names of pattern-degenerate phases this candidate absorbed —
    # same space group + lattice (within 2%), so EBSD can't distinguish
    # them. Empty for a unique structure. UI shows "also matches: …".
    degenerate_with: list[str] = []


class IndexedPhaseOut(BaseModel):
    """Per-pixel indexed phase info, populated when an indexing result is
    active. Lets the UI show "you're sitting on phase X — symmetry agrees/
    doesn't agree" inline with the Crystal Hint analysis."""
    phase_id: int
    phase_name: str = ""
    crystal_system: str = "unknown"
    space_group: str = ""
    is_unindexed: bool = False
    # Whether the symmetry-detected compatible_systems include the indexed
    # phase's crystal_system — quick "consistent / mismatch" badge.
    consistent_with_symmetry: Optional[bool] = None


class AnalyzePixelResponse(BaseModel):
    pixel_index: int
    symmetry: dict           # SymmetryReport.to_dict()
    lattice: dict            # LatticeReport.to_dict()
    local_matches: list[CandidateOut]
    external_pending: bool = False
    preset_key: Optional[str] = None
    warnings: list[str] = []
    pattern_b64_png: Optional[str] = None    # base64 PNG of the experimental pattern (for UI preview)
    pattern_shape: Optional[list[int]] = None
    indexed_phase: Optional[IndexedPhaseOut] = None


def _local_match_to_out(m: LocalMatch, preset_key: Optional[str]) -> CandidateOut:
    """Convert a LocalMatch to API output, boosting score if the match is in
    the preset's expected_phases.

    We try BOTH name-token matching (works for `Al`, `Si`, `Al2Cu`-style keys)
    AND structural matching on (lattice ± 8%, same crystal_system) — needed
    because sd_*-prefixed entries (e.g. `sd_0302719` = α-AlFeSi Im-3, a=12.5)
    don't share name tokens with the preset's "alpha-Al(Fe,Mn)Si" string.
    """
    expected_in_preset = False
    boosted_score = m.score
    # Scale the preset boost by how well the pixel's measured chemistry
    # actually supports this phase. A preset is a PRIOR (these phases are
    # expected SOMEWHERE in the alloy), not a per-pixel verdict — so a fixed
    # ×1.5 let the matrix phase (Al) override the measured chemistry and win
    # even on an intermetallic-particle pixel (the user's complaint: "Al is
    # always top even where we have an intermetallic"). Tie the boost to
    # chemistry_fit: full ×1.5 when the chemistry agrees, tapering to ×1.0
    # (no boost) when the pixel chemistry contradicts the phase. When EDS is
    # unavailable (chemistry_fit is None — e.g. EDAX/Ni data) keep the full
    # ×1.5 so non-EDS behaviour is unchanged.
    _boost = (1.0 + 0.5 * m.chemistry_fit) if m.chemistry_fit is not None else 1.5
    if preset_key:
        try:
            # 1) name-token match
            boosted = boost_score_for_expected_phase(
                preset_key, m.entry.key, base_score=m.score, boost=_boost
            )
            if boosted != m.score:
                expected_in_preset = True
                boosted_score = boosted
            else:
                # 2) structural match: same system + lattice within ~8% of any
                #    expected phase in the preset
                from backend.api.services.crystal_hint_presets import get_preset
                preset = get_preset(preset_key)
                for ph in preset.expected_phases:
                    if m.entry.crystal_system != "unknown" and m.entry.lattice_a_A:
                        if (abs(m.entry.lattice_a_A - ph.a_A) / max(ph.a_A, 1e-3) < 0.08):
                            # Also require space-group match if both known
                            same_sg = (
                                not ph.sg or not m.entry.space_group
                                or m.entry.space_group.replace(" ", "").lower()
                                   == ph.sg.replace(" ", "").lower()
                            )
                            if same_sg:
                                expected_in_preset = True
                                boosted_score = m.score * _boost
                                break
        except KeyError:
            pass
    return CandidateOut(
        key=m.entry.key,
        formula=m.entry.formula,
        display_formula=m.entry.display_formula or m.entry.formula,
        space_group=m.entry.space_group,
        space_group_number=m.entry.space_group_number,
        crystal_system=m.entry.crystal_system,
        elements=list(m.entry.elements),
        lattice_a_A=m.entry.lattice_a_A,
        lattice_b_A=m.entry.lattice_b_A,
        lattice_c_A=m.entry.lattice_c_A,
        n_atoms=m.entry.n_atoms,
        has_sht=m.has_sht,
        plausibility=m.plausibility,
        lattice_distance=m.lattice_distance,
        system_match=m.system_match,
        score=boosted_score,
        expected_in_preset=expected_in_preset,
        symmetry_fit=m.symmetry_fit,
        dspacing_fit=m.dspacing_fit,
        chemistry_fit=m.chemistry_fit,
    )


def _get_pixel_at_pct(pixel_index: int) -> Optional[dict[str, float]]:
    """Delegates to the shared eds_pixel_chemistry service (kept as a thin
    wrapper so existing call sites in this module need no change)."""
    from backend.api.services.eds_pixel_chemistry import get_pixel_at_pct
    return get_pixel_at_pct(pixel_index)


@router.post("/analyze-pixel", response_model=AnalyzePixelResponse)
def analyze_pixel(req: AnalyzePixelRequest):
    """Mode 1 (Single Pixel Deep): symmetry + lattice + ranked candidates.

    Pulls the pattern from the active h5_session.
    Returns IMMEDIATELY with local matches; external DB lookup not in v1
    (external_pending=False).
    """
    if not h5_session.is_open():
        raise HTTPException(status_code=400, detail="No EBSD file loaded — open one first")

    # Get pattern. PREFER the active EBSD-viewer signal (which is the
    # currently-selected dataset — could be the CLAHE'd derivative, BG-
    # removed, etc.). The user expects "what they see in the EBSD viewer
    # is what gets analysed". Falls back to h5_session for the raw file
    # only if no active signal is mounted.
    pattern = _fetch_active_pattern(req.pixel_index)
    if pattern is None:
        try:
            pattern = h5_session.get_cached_pattern(req.pixel_index, pattern_type=req.pattern_type)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load pattern: {exc}") from exc
    if pattern is None:
        raise HTTPException(status_code=404,
                             detail=f"Pattern at index {req.pixel_index} not available")

    pattern_arr = np.asarray(pattern, dtype=np.float32)
    if pattern_arr.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Pattern at index {req.pixel_index} is not 2D (shape={pattern_arr.shape})",
        )

    # 1. Detector geometry — needed by both symmetry (Method B) and lattice
    geom = _gather_detector_geom()

    # 2. Symmetry detection
    # Prefer Method B (spherical projection) when detector geometry is
    # available — it works correctly for off-center pattern centers and
    # tilted detectors, where Method A (image-space rotation) under-scores
    # by ~3-5x on real raw patterns. Fall back to Method A if geometry is
    # missing. Optional neighborhood averaging suppresses per-pixel noise
    # and roughly doubles the NCC. The chosen method is reflected in
    # sym_report.method (returned via the symmetry payload).
    # Compute the analysis pattern: either the single pixel (default) or
    # an unit-mean-normalised neighborhood average (if avg_radius > 0).
    # The same averaged pattern is reused for lattice estimation below so
    # both stages benefit from the noise reduction.
    analysis_pattern = pattern_arr
    n_averaged = 1
    if req.avg_radius > 0:
        try:
            neighbours = _collect_neighbour_patterns(
                req.pixel_index, req.avg_radius, req.pattern_type,
            )
            if len(neighbours) > 1:
                # Unit-mean normalise each pattern before averaging so a
                # single bright outlier doesn't dominate (matches the logic
                # in detect_symmetry_spherical_averaged).
                normed = []
                for p in neighbours:
                    m = float(p.mean())
                    normed.append(p / m if m > 1e-6 else p)
                analysis_pattern = np.mean(normed, axis=0).astype(np.float32)
                n_averaged = len(neighbours)
        except Exception as exc:
            logger.warning("Neighborhood averaging failed: %s — falling back "
                           "to single pixel", exc)

    try:
        if geom is not None and n_averaged > 1:
            # Re-use the analysis_pattern we just computed — call
            # detect_symmetry_spherical_averaged with a single pre-averaged
            # pattern to keep the API contract (method tag stays correct).
            sym_report = detect_symmetry_spherical(analysis_pattern, geom)
            sym_report.method = f"spherical_avg{n_averaged}"
            sym_report.n_patterns_averaged = n_averaged
        elif geom is not None:
            sym_report = detect_symmetry_spherical(analysis_pattern, geom)
        else:
            sym_report = detect_symmetry(analysis_pattern)
    except Exception as exc:
        logger.exception("Symmetry detection failed (geom=%s)", geom is not None)
        # If spherical failed, try the image-space fallback once before
        # giving up — we'd rather return an imperfect result than 500.
        if geom is not None:
            try:
                sym_report = detect_symmetry(pattern_arr)
                sym_report.method = "image_fallback"
            except Exception as exc2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Symmetry detection failed: {exc2}",
                ) from exc2
        else:
            raise HTTPException(
                status_code=500, detail=f"Symmetry detection failed: {exc}",
            ) from exc

    # 3. Lattice estimation — need detector geometry
    if geom is None:
        # Can't estimate lattice without geometry, but symmetry is still valuable
        lattice_report = LatticeReport(warnings=["No detector geometry available — lattice estimation skipped"])
    else:
        # Use the best-confidence inferred system as hint
        sys_hint = None
        if sym_report.compatible_systems:
            for candidate in ("cubic", "hexagonal", "tetragonal"):
                if candidate in sym_report.compatible_systems:
                    sys_hint = candidate
                    break
        try:
            lattice_report = estimate_lattice(analysis_pattern, geom, crystal_system_hint=sys_hint)
        except Exception as exc:
            logger.exception("Lattice estimation failed")
            lattice_report = LatticeReport(warnings=[f"lattice estimation error: {exc}"])

    # 4. Local library search
    # Determine system filter
    system_filter = req.crystal_system_filter
    if system_filter is None and sym_report.compatible_systems:
        # Use the first compatible system as a soft filter
        # (cubic is preferred if listed; else first available)
        for candidate in ("cubic", "hexagonal", "tetragonal", "orthorhombic"):
            if candidate in sym_report.compatible_systems:
                system_filter = candidate
                break

    # Determine lattice range
    a_range = None
    if lattice_report.a_range_A:
        a_range = tuple(lattice_report.a_range_A)

    # Pass the FULL compatible_systems set so phases with non-best-fold
    # symmetries (e.g. tetragonal with 4-fold while 3-fold ranked higher)
    # don't get penalised as off-system. Without this, Al7FeCu2 (tetragonal)
    # was wrongly flagged off-system on a 3-fold-dominant pattern, even
    # though the 4-fold NCC was also strong.
    # Pass n_fold_scores + d_observed so the library scorer can apply the
    # per-phase symmetry_fit + dspacing_fit multipliers. Without these,
    # every constraint-passing phase tied at the same boosted score.
    _n_fold_for_fit = {
        int(k): float(v) for k, v in sym_report.n_fold_scores.items()
    } if sym_report.n_fold_scores else None
    _d_obs_for_fit = lattice_report.d_spacings_A or None
    # The d-spacing fit only helps when the Hough-derived d-spacings are
    # real. On these patterns the detector reports confidence="none" (band
    # detection dominated by the ±45° frame), so gate the fit on at-least-
    # "low" confidence to keep noise out of the ranking.
    _d_obs_trustworthy = lattice_report.confidence in ("low", "medium", "high")
    # EDS chemistry weighting: resolve the pixel's measured At% from the
    # active dataset. None (no EDS) → weighting silently degrades to "off".
    _pixel_at_pct = None
    _eds_mode = req.eds_weighting
    if _eds_mode in ("soft", "filter"):
        _pixel_at_pct = _get_pixel_at_pct(req.pixel_index)
        if not _pixel_at_pct:
            _eds_mode = "off"  # no EDS on this dataset — fail soft
    _eds_stats: dict = {}
    local_matches = library_search(
        elements=req.elements,
        crystal_system=system_filter,
        compatible_systems=sym_report.compatible_systems or None,
        a_range_A=a_range,
        strict_chemistry=req.strict_chemistry,
        n_fold_scores=_n_fold_for_fit,
        d_observed_A=_d_obs_for_fit,
        d_observed_trustworthy=_d_obs_trustworthy,
        pixel_at_pct=_pixel_at_pct,
        eds_weighting=_eds_mode,
        eds_filter_threshold=req.eds_filter_threshold,
        stats=_eds_stats,
    )

    # 5. Apply preset boost + convert to response model
    if req.preset_key:
        # Map preset elements onto sample if user didn't provide elements
        if not req.elements:
            try:
                preset = get_preset(req.preset_key)
                local_matches = library_search(
                    elements=list(preset.elements),
                    crystal_system=system_filter,
                    compatible_systems=sym_report.compatible_systems or None,
                    a_range_A=a_range,
                    strict_chemistry=req.strict_chemistry,
                    n_fold_scores=_n_fold_for_fit,
                    d_observed_A=_d_obs_for_fit,
                    d_observed_trustworthy=_d_obs_trustworthy,
                    pixel_at_pct=_pixel_at_pct,
                    eds_weighting=_eds_mode,
                    eds_filter_threshold=req.eds_filter_threshold,
                    stats=_eds_stats,
                )
            except KeyError:
                pass

    # Boost scores for preset-expected phases FIRST so the truncation doesn't
    # drop a candidate that would have ranked into the top after boost.
    candidates_out = [_local_match_to_out(m, req.preset_key) for m in local_matches]
    # Sort by:
    #   1. -score (highest first — preset-expected boosted to 1.5)
    #   2. lattice_distance (closer to target lattice wins)
    #   3. -has_sht (SHT-available before SHT-missing)
    #   4. n_atoms (simpler structures first — Al's 4 atoms before
    #      α-Al(Fe,Mn)Si's 150 atoms. Matrix-like phases bubble up,
    #      which on Al-alloy scans is what the user almost always wants
    #      to see ranked first. Without this, alphabetical tied-score
    #      ordering put complex polymorphs ahead of simple matrix phases.)
    #   5. key (stable alphabetical on full tie)
    candidates_out.sort(key=lambda c: (
        -c.score,
        c.lattice_distance,
        0 if c.has_sht else 1,
        c.n_atoms or 999,   # None → bottom
        c.key,
    ))

    # Collapse pattern-degenerate candidates: phases with the same space
    # group + lattice (within 2%) produce indistinguishable Kikuchi
    # patterns, so they shouldn't each occupy a top slot. Keep the
    # highest-scoring representative (list is already score-sorted) and
    # record the absorbed names on degenerate_with. This frees slots for
    # genuinely distinct structures — directly lifting the phases that were
    # stuck behind near-duplicate cubic α-Al(Fe,Mn) entries.
    from backend.api.services.crystal_hint_phase_fit import collapse_degenerate
    _deg_items = [{
        "name": c.display_formula or c.formula,
        "crystal_system": c.crystal_system,
        "space_group": c.space_group,
        "sg_number": c.space_group_number,
        "a": c.lattice_a_A, "b": c.lattice_b_A, "c": c.lattice_c_A,
        "_obj": c,
    } for c in candidates_out]
    _reps = collapse_degenerate(_deg_items)
    candidates_out = []
    for rep in _reps:
        obj = rep["_obj"]
        obj.degenerate_with = rep["degenerate_with"]
        candidates_out.append(obj)
    candidates_out = candidates_out[:20]

    warnings = list(sym_report.warnings) + list(lattice_report.warnings)
    # The symmetry method is exposed as a proper field on the symmetry
    # report (sym_report.method); no need to also stuff it into warnings.
    if req.eds_weighting in ("soft", "filter") and _eds_mode == "off":
        warnings.append(
            "EDS chemistry weighting requested but the active dataset has no "
            "EDS data — ranking used pattern symmetry/lattice only.")
    _n_filtered = _eds_stats.get("eds_filtered", 0)
    if _n_filtered:
        warnings.append(
            f"EDS filter dropped {_n_filtered} phase(s) with chemistry match "
            f"below {req.eds_filter_threshold:.2f} — switch EDS weighting to "
            f"'soft' to see them ranked lower instead of removed.")

    # Encode pattern as base64 PNG for the UI preview (gray colormap).
    # We send it back in the same response so the frontend can show the
    # pattern alongside the symmetry/lattice analysis without a second call.
    try:
        # Log pattern stats so we can debug "pattern preview is blank" issues
        # — if the array is all-zeros or all-NaN the user will see a white
        # box and we want a trace to point at it.
        pmin = float(np.nanmin(pattern_arr))
        pmax = float(np.nanmax(pattern_arr))
        pmean = float(np.nanmean(pattern_arr))
        logger.info(
            "crystal_hint: pattern[%d] stats min=%.3f max=%.3f mean=%.3f shape=%s",
            req.pixel_index, pmin, pmax, pmean, pattern_arr.shape,
        )
        if pmax - pmin < 1e-6:
            logger.warning(
                "crystal_hint: pattern[%d] is uniform (min==max=%.3f) — "
                "preview will be flat. Check h5_session.get_cached_pattern.",
                req.pixel_index, pmin,
            )
        # array_to_base64_raw is ~30x faster than matplotlib (PIL direct
        # path, ~2ms vs ~60ms on a 118x118 pattern). For a plain greyscale
        # EBSD pattern we don't need a colormap, so the raw path is fine.
        pattern_png_b64 = array_to_base64_raw(pattern_arr)
        pattern_shape = list(pattern_arr.shape)
    except Exception as exc:
        logger.warning("Pattern PNG encoding failed: %s", exc)
        pattern_png_b64 = None
        pattern_shape = list(pattern_arr.shape)

    # 6. Indexed-phase lookup. If an indexing result is active, report
    #    which phase this pixel was indexed as and whether the symmetry
    #    detection agrees. Lets the UI render a "consistent / mismatch"
    #    badge inline. Cheap (just an xmap index lookup); on failure the
    #    field stays None and the UI hides the badge.
    indexed_phase: Optional[IndexedPhaseOut] = None
    try:
        xmap, _result = _get_active_xmap()
        if xmap is not None:
            indexed_phase = _resolve_indexed_phase(
                xmap, req.pixel_index, sym_report.compatible_systems,
            )
    except Exception as exc:
        logger.debug("indexed-phase lookup failed for pixel %d: %s",
                     req.pixel_index, exc)

    return AnalyzePixelResponse(
        pixel_index=req.pixel_index,
        symmetry=sym_report.to_dict(),
        lattice=lattice_report.to_dict(),
        local_matches=candidates_out,
        external_pending=False,
        preset_key=req.preset_key,
        warnings=warnings,
        pattern_b64_png=pattern_png_b64,
        pattern_shape=pattern_shape,
        indexed_phase=indexed_phase,
    )


def _resolve_indexed_phase(
    xmap, pixel_index: int, detected_systems: list[str],
) -> Optional[IndexedPhaseOut]:
    """Resolve the indexed phase for one pixel + check symmetry consistency.

    Returns None if the xmap doesn't contain this pixel index. -1 phase_id
    is the orix sentinel for "unindexed" → reported as is_unindexed=True.

    Assumes `xmap.phase_id` is a flat (1-D) array of length n_pixels —
    consistent with how the indexing route builds it. Out-of-range indices
    return None (we'd rather hide the badge than show wrong phase info).
    """
    # Explicit bounds check before indexing — list[OOB] would silently
    # return empty, and np.ndarray[OOB] raises IndexError. Guard against both.
    try:
        n = len(xmap.phase_id)
    except Exception:
        return None
    if not (0 <= pixel_index < n):
        return None
    try:
        phase_id = int(xmap.phase_id[pixel_index])
    except (IndexError, AttributeError, TypeError):
        return None
    if phase_id < 0:
        return IndexedPhaseOut(
            phase_id=-1, phase_name="", crystal_system="unknown",
            is_unindexed=True, consistent_with_symmetry=None,
        )
    # Phase lookup: orix PhaseList supports both dict-style get and
    # attribute access on real objects; tests sometimes pass a Mock.
    phase_name = ""
    crystal_system = "unknown"
    space_group = ""
    try:
        phase = xmap.phases[phase_id]
        phase_name = str(getattr(phase, "name", "") or "")
        ps = getattr(phase, "point_group", None)
        if ps is not None:
            sym = getattr(ps, "name", "")
            # Heuristic mapping point-group → crystal-system via space group
            sg = getattr(phase, "space_group", None)
            if sg is not None:
                space_group = str(getattr(sg, "short_name", "") or "")
                sg_num = getattr(sg, "number", None)
                if sg_num is not None:
                    crystal_system = _system_from_sg_number_local(int(sg_num))
    except Exception:
        pass

    # Consistency check: does the detected n-fold's compatible_systems
    # include the indexed phase's crystal_system?
    consistent: Optional[bool] = None
    if detected_systems and crystal_system != "unknown":
        # Treat trigonal/hexagonal as compatible (rhombohedral lives on
        # the boundary between the two in many EBSD heuristics).
        equiv = {"trigonal", "hexagonal"}
        if crystal_system in equiv:
            consistent = bool({s for s in detected_systems} & equiv)
        else:
            consistent = crystal_system in detected_systems

    return IndexedPhaseOut(
        phase_id=phase_id,
        phase_name=phase_name,
        crystal_system=crystal_system,
        space_group=space_group,
        is_unindexed=False,
        consistent_with_symmetry=consistent,
    )


def _system_from_sg_number_local(sg: int) -> str:
    if 1 <= sg <= 2: return "triclinic"
    if 3 <= sg <= 15: return "monoclinic"
    if 16 <= sg <= 74: return "orthorhombic"
    if 75 <= sg <= 142: return "tetragonal"
    if 143 <= sg <= 167: return "trigonal"
    if 168 <= sg <= 194: return "hexagonal"
    if 195 <= sg <= 230: return "cubic"
    return "unknown"


def _fetch_active_pattern(pixel_index: int) -> Optional[np.ndarray]:
    """Fetch a pattern from the EBSD-viewer's ACTIVE dataset (which may be a
    CLAHE'd or background-removed derivative). Returns None if no active
    signal is mounted — caller should fall back to h5_session for the raw
    file.

    The EBSD Viewer state lives in `backend.api.routes.ebsd_viewer`. We
    access it dynamically via module attributes (not import-time bindings)
    because `_active_dataset` and `_raw_signals` mutate over the session.
    """
    try:
        from backend.api.routes import ebsd_viewer as _ev
        signal = _ev._get_active_signal()
        if signal is None:
            return None
        nav = signal.axes_manager.navigation_shape
        n_cols = int(nav[0]) if len(nav) >= 1 else 1
        n_rows = int(nav[1]) if len(nav) >= 2 else 1
        if not (0 <= pixel_index < n_rows * n_cols):
            return None
        row, col = divmod(pixel_index, n_cols)
        pat = signal.data[row, col]
        # Materialise dask-backed lazy arrays before downstream NumPy work
        if hasattr(pat, "compute"):
            pat = pat.compute()
        arr = np.asarray(pat, dtype=np.float32)
        # Sanity: only accept a real 2-D pattern. Tests sometimes patch
        # `_get_active_signal` with a MagicMock that returns a 0-d mock
        # — fall through to h5_session in that case rather than raising.
        if arr.ndim != 2 or arr.size == 0:
            return None
        return arr
    except Exception as exc:
        logger.debug("active-pattern fetch failed for %d: %s", pixel_index, exc)
        return None


def _get_scan_dimensions() -> Optional[tuple[int, int]]:
    """Return (n_rows, n_cols) for the active scan, or None if not loadable.

    The H5OINADataExtractor exposes `get_grid_dimensions()`; tests with
    MagicMock often patch `get_metadata()` instead, so we try both APIs.
    """
    # The pixel indices this feeds (neighbourhood gathering, ROI masks) are
    # flat indices over the ACTIVE dataset's grid, which is the cropped grid
    # when the user is working on a cut-out. Off the crop path
    # get_active_extractor() returns the raw extractor unchanged.
    ext = h5_session.get_active_extractor()
    if ext is None:
        return None
    if hasattr(ext, "get_grid_dimensions"):
        try:
            n_rows, n_cols = ext.get_grid_dimensions()
            return int(n_rows), int(n_cols)
        except Exception:
            pass
    if hasattr(ext, "get_metadata"):
        try:
            meta = ext.get_metadata()
            n_rows = (meta.get("n_rows") or meta.get("nrows")
                      or meta.get("scan_shape", [None])[0])
            n_cols = (meta.get("n_cols") or meta.get("ncols")
                      or meta.get("scan_shape", [None, None])[-1])
            if n_rows is not None and n_cols is not None:
                return int(n_rows), int(n_cols)
        except Exception:
            pass
    return None


def _collect_neighbour_patterns(
    center_idx: int,
    radius: int,
    pattern_type: str = "processed",
) -> list[np.ndarray]:
    """Collect a (2*radius+1) × (2*radius+1) neighborhood of patterns around
    `center_idx` in the active EBSD scan, for symmetry averaging.

    Falls back gracefully:
      - patterns near the scan edge return as many in-bounds neighbours as
        exist (typically 4-6 instead of 9 for radius=1).
      - if the center pattern itself fails to load, returns an empty list
        (caller should treat as "no patterns").
      - the center pattern itself is always included if loadable.
    """
    if radius <= 0:
        try:
            p = h5_session.get_cached_pattern(center_idx, pattern_type=pattern_type)
            return [np.asarray(p, dtype=np.float32)] if p is not None else []
        except Exception:
            return []

    dims = _get_scan_dimensions()
    if dims is None:
        # Without scan shape we can't do 2-D neighborhoods. Fall back to a
        # 1-D window in flat-index space (less ideal but still useful).
        idxs = [center_idx + d for d in range(-radius, radius + 1)
                if center_idx + d >= 0]
    else:
        n_rows, n_cols = dims
        r0, c0 = divmod(center_idx, n_cols)
        idxs = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    idxs.append(r * n_cols + c)

    patterns: list[np.ndarray] = []
    for idx in idxs:
        # Same preference as `_fetch_active_pattern`: active EBSD-viewer
        # signal (CLAHE'd/processed) → h5_session (raw file). The user
        # expects neighborhood averaging to use the SAME pattern source
        # they see in the EBSD viewer.
        p = _fetch_active_pattern(idx)
        if p is None:
            try:
                p = h5_session.get_cached_pattern(idx, pattern_type=pattern_type)
            except Exception:
                continue
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim == 2:
            patterns.append(arr)
    return patterns


def _gather_detector_geom() -> Optional[dict]:
    """Assemble detector geometry from the active EBSD signal.

    Reads from the same sources the rest of the app uses (calibration_store
    → fallback to signal.detector) so we always agree with what the user
    sees in the Indexing / PC Refinement / Phase Map pages.

    Returns dict with pc_x, pc_y, pc_z, pat_width, pat_height, voltage_kv
    or None if no signal is loaded / detector unavailable.

    Voltage handling: if the H5 file's voltage header is missing or
    corrupt (the EDAX file format sometimes stores 7.3e-24), falls back
    to 20 kV with a logged warning rather than failing loud — Crystal
    Hint's lattice estimator already warns on corrupt voltage in its
    own report, so we don't need to fail twice.
    """
    try:
        # Access ebsd_viewer state via the MODULE, not direct value imports:
        # `_active_dataset` and `_ebsd_file_path` are module-level variables
        # that get mutated when the user loads a file. Capturing them at
        # import time via `from ... import varname` would bind to the
        # initial None values forever. Module attribute lookup IS dynamic.
        from backend.api.routes import ebsd_viewer as _ev
        # Import the singleton (NOT the module). The module also contains a
        # variable named `calibration_store` which is the singleton instance;
        # ebsd_viewer.py imports it the same way.
        from backend.api.services.calibration_store import calibration_store

        signal = _ev._get_active_signal()
        if signal is None:
            return None

        det = calibration_store.get_detector(_ev._active_dataset)
        if det is None:
            det = getattr(signal, "detector", None)
        if det is None:
            return None

        # det.pc is a list/ndarray [PCx, PCy, PCz] in Bruker convention.
        # For per-pixel PC datasets it may be 2-D (n_patterns × 3); flatten
        # and take the first triplet either way.
        pc = getattr(det, "pc", None)
        if pc is None:
            return None
        import numpy as _np
        pc_arr = _np.asarray(pc, dtype=float).reshape(-1)
        if pc_arr.size < 3:
            return None
        # Pattern shape from signal axes manager
        sig_shape = signal.axes_manager.signal_shape
        pat_w = int(sig_shape[0]) if sig_shape else None
        pat_h = int(sig_shape[1]) if len(sig_shape) >= 2 else pat_w

        # Voltage: try H5 file, fall back to 20 kV
        voltage = None
        try:
            if _ev._ebsd_file_path:
                voltage = _ev._extract_beam_energy(_ev._ebsd_file_path)
        except Exception:
            voltage = None
        if voltage is None or not (1.0 < float(voltage) < 100.0):
            logger.info("crystal_hint: voltage missing/corrupt — defaulting to 20 kV")
            voltage = 20.0

        return {
            # Use pc_arr (flattened) — `pc` may be 2-D per-pixel, in which
            # case pc[0] is a 3-vector and float() raises TypeError, getting
            # silently caught by the outer except → "geom is None" downstream,
            # Method B falls back to Method A, lattice estimation skipped.
            "pc_x": float(pc_arr[0]),
            "pc_y": float(pc_arr[1]),
            "pc_z": float(pc_arr[2]),
            "pat_width": pat_w or 118,
            "pat_height": pat_h or pat_w or 118,
            "voltage_kv": float(voltage),
        }
    except Exception as exc:
        logger.warning("crystal_hint: could not gather detector geom: %s", exc)
        return None


@router.get("/presets")
def list_presets():
    """Return all material presets (UI dropdown)."""
    presets = get_presets()
    return {
        key: {
            "label": p.label,
            "description": p.description,
            "elements": list(p.elements),
            "expected_phases": [
                {
                    "name": ph.name, "sg": ph.sg, "a_A": ph.a_A, "c_A": ph.c_A,
                    "category": ph.category,
                }
                for ph in p.expected_phases
            ],
            "heat_treatments": list(p.heat_treatments),
        }
        for key, p in presets.items()
    }


@router.get("/presets/{key}")
def get_preset_endpoint(key: str):
    """Get one preset by key."""
    try:
        p = get_preset(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "key": p.key,
        "label": p.label,
        "description": p.description,
        "elements": list(p.elements),
        "expected_phases": [
            {
                "name": ph.name, "sg": ph.sg, "a_A": ph.a_A, "c_A": ph.c_A,
                "category": ph.category,
            }
            for ph in p.expected_phases
        ],
        "heat_treatments": list(p.heat_treatments),
    }


class AnalyzeRegionRequest(BaseModel):
    """Region/Whole-Scan analysis input (Mode 2)."""

    elements: list[str] = Field(default_factory=list)
    preset_key: Optional[str] = None
    # ROI shape — None / "whole" means full scan.
    roi_type: str = Field("whole", description="'whole' | 'rect' | 'polygon'")
    # For 'rect': r_min, r_max, c_min, c_max (inclusive bounds)
    rect: Optional[list[int]] = Field(None, description="[r_min, r_max, c_min, c_max]")
    # For 'polygon': list of [row, col] vertices defining a closed polygon
    polygon: Optional[list[list[int]]] = None
    max_pixels: int = Field(256, ge=4, le=4096,
                             description="Cap on number of pixels analyzed (subsampled if larger).")
    avg_radius: int = Field(0, ge=0, le=3,
                            description="Per-pixel neighborhood averaging — same as Mode 1's "
                            "avg_radius. 0=single, 1=3x3, 2=5x5. Suppresses noise but takes "
                            "(2*r+1)^2 more pattern fetches per analysed pixel.")


class AnalyzeRegionResponse(BaseModel):
    n_pixels_analyzed: int
    n_pixels_in_roi: int
    n_pixels_no_symmetry: int
    n_fold_histogram: dict           # str(int)→count
    crystal_system_histogram: dict
    lattice_a_histogram: dict
    library_match_rate: float
    missing_phase_clusters: list[dict]
    elapsed_seconds: float
    warnings: list[str] = []
    preset_key: Optional[str] = None


@router.post("/analyze-region", response_model=AnalyzeRegionResponse)
def analyze_region_endpoint(req: AnalyzeRegionRequest):
    """Mode 2 (Region / Whole-Scan): statistical analysis across many pixels.

    Reports symmetry distribution, lattice histogram, library-match rate,
    and clusters of "missing phase" signatures (pixel groups that share
    crystal-system + lattice but have NO local library match).
    """
    if not h5_session.is_open():
        raise HTTPException(status_code=400, detail="No EBSD file loaded — open one first")

    if h5_session.get_active_extractor() is None:
        raise HTTPException(status_code=500, detail="EBSD extractor not available")

    dims = _get_scan_dimensions()
    if dims is None:
        raise HTTPException(
            status_code=500,
            detail="Cannot determine scan shape from extractor "
                   "(needs get_grid_dimensions() or get_metadata()).",
        )
    n_rows, n_cols = dims

    # Build ROI mask from request
    roi_mask = _build_roi_mask(n_rows, n_cols, req.roi_type, req.rect, req.polygon)

    # Effective elements list (preset fallback)
    elements = list(req.elements)
    if not elements and req.preset_key:
        try:
            preset = get_preset(req.preset_key)
            elements = list(preset.elements)
        except KeyError:
            pass

    detector_geom = _gather_detector_geom()

    def _get_pattern(row: int, col: int):
        # Convert (row, col) → flat index and reuse get_cached_pattern
        idx = row * n_cols + col
        return h5_session.get_cached_pattern(idx, pattern_type="processed")

    result = analyze_region(
        get_pattern=_get_pattern,
        n_rows=n_rows,
        n_cols=n_cols,
        roi_mask=roi_mask,
        detector_geom=detector_geom,
        elements=elements,
        max_pixels=req.max_pixels,
        avg_radius=req.avg_radius,
    )

    return AnalyzeRegionResponse(
        **result.to_dict(),
        preset_key=req.preset_key,
    )


def _build_roi_mask(
    n_rows: int,
    n_cols: int,
    roi_type: str,
    rect: Optional[list[int]],
    polygon: Optional[list[list[int]]],
) -> Optional[np.ndarray]:
    """Convert ROI request → boolean mask (n_rows, n_cols), or None for whole."""
    rt = (roi_type or "whole").lower()
    if rt == "whole":
        return None
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    if rt == "rect" and rect and len(rect) == 4:
        r_min, r_max, c_min, c_max = rect
        # Fail loud on inverted bounds — otherwise the user thinks they
        # selected a small ROI but the backend silently falls back to whole
        # scan. See code-review B6.
        if r_min > r_max or c_min > c_max:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ROI rect has inverted bounds: "
                    f"[r_min={r_min}, r_max={r_max}, c_min={c_min}, c_max={c_max}]. "
                    "Expected r_min ≤ r_max and c_min ≤ c_max."
                ),
            )
        r_min = max(0, int(r_min)); r_max = min(n_rows - 1, int(r_max))
        c_min = max(0, int(c_min)); c_max = min(n_cols - 1, int(c_max))
        mask[r_min:r_max + 1, c_min:c_max + 1] = True
        return mask
    if rt == "polygon" and polygon and len(polygon) >= 3:
        try:
            from matplotlib.path import Path
        except ImportError:
            return None
        # polygon vertices = [[row, col], ...] — convert to (col, row) for Path
        verts = [(float(c), float(r)) for r, c in polygon]
        path = Path(verts)
        ys, xs = np.mgrid[:n_rows, :n_cols]
        points = np.column_stack([xs.ravel(), ys.ravel()])
        inside = path.contains_points(points).reshape(n_rows, n_cols)
        return inside
    return None


class QualityCheckRequest(BaseModel):
    """Mode 3 input — compare detected pattern symmetry vs indexed phase."""

    roi_type: str = Field("whole", description="'whole' | 'rect' | 'polygon'")
    rect: Optional[list[int]] = None
    polygon: Optional[list[list[int]]] = None
    max_pixels: int = Field(256, ge=4, le=4096)
    avg_radius: int = Field(0, ge=0, le=3,
                            description="Neighborhood averaging — same as Mode 1/2.")


class QualityCheckResponse(BaseModel):
    n_pixels_analyzed: int
    n_pixels_in_roi: int
    n_pixels_indexed: int
    n_pixels_unindexed: int
    n_pixels_match: int
    n_pixels_mismatch: int
    n_pixels_no_detection: int
    mismatch_rate: float
    mismatch_by_phase: dict
    sample_mismatches: list[dict] = []
    elapsed_seconds: float
    warnings: list[str] = []


def _get_active_xmap():
    """Pull the active indexing-result xmap, or None if none active."""
    try:
        # The indexing route module owns _result_registry / _active_result_id.
        # Import lazily to avoid circular imports at module load.
        from backend.api.routes import indexing as indexing_module

        result = indexing_module._get_result(None)
        if result is None:
            return None, None
        xmap = getattr(result, "xmap", None)
        return xmap, result
    except Exception as exc:
        logger.warning("Failed to load active indexing result: %s", exc)
        return None, None


@router.post("/quality-check", response_model=QualityCheckResponse)
def quality_check_endpoint(req: QualityCheckRequest):
    """Mode 3: compare detected pattern symmetry vs indexed phase symmetry.

    Requires an active indexing result (run /api/indexing first).
    Returns counts of match / mismatch / unindexed / no-detection pixels
    plus per-indexed-phase breakdown of which phase is most often
    mis-indexed.
    """
    if not h5_session.is_open():
        raise HTTPException(status_code=400, detail="No EBSD file loaded — open one first")
    xmap, _result = _get_active_xmap()
    if xmap is None:
        raise HTTPException(
            status_code=400,
            detail="No active indexing result — run /api/indexing first.",
        )

    if h5_session.get_active_extractor() is None:
        raise HTTPException(status_code=500, detail="EBSD extractor not available")
    dims = _get_scan_dimensions()
    if dims is None:
        raise HTTPException(status_code=500,
                            detail="Cannot determine scan shape.")
    n_rows, n_cols = dims

    roi_mask = _build_roi_mask(n_rows, n_cols, req.roi_type, req.rect, req.polygon)

    def _get_pattern(row: int, col: int):
        idx = row * n_cols + col
        return h5_session.get_cached_pattern(idx, pattern_type="processed")

    qresult = quality_check_region(
        get_pattern=_get_pattern,
        xmap=xmap,
        n_rows=n_rows, n_cols=n_cols,
        roi_mask=roi_mask,
        max_pixels=req.max_pixels,
        detector_geom=_gather_detector_geom(),
        avg_radius=req.avg_radius,
    )
    return QualityCheckResponse(**qresult.to_dict())


class ExternalSearchRequest(BaseModel):
    elements: list[str] = Field(default_factory=list)
    crystal_system: Optional[str] = Field(None,
        description="Filter for cubic/hexagonal/tetragonal/etc.")
    a_low_A: Optional[float] = Field(None, gt=0, description="Lower lattice bound")
    a_high_A: Optional[float] = Field(None, gt=0, description="Upper lattice bound")
    max_results: int = Field(20, ge=1, le=100)
    # Optional pattern-derived hints for per-phase fit scoring. When the
    # frontend passes the analyze-pixel result's n_fold_scores + lattice
    # d_spacings here, each external candidate gets the same
    # symmetry_fit + dspacing_fit treatment as local matches and the
    # list is resorted by combined fit instead of just lattice distance.
    n_fold_scores: Optional[dict[str, float]] = Field(None,
        description="Per-fold NCC from the symmetry detector (str keys "
                    "for JSON compat: '2', '3', '4', '6').")
    d_observed_A: Optional[list[float]] = Field(None,
        description="Observed Hough-band d-spacings in Å.")
    d_observed_trustworthy: bool = Field(False,
        description="Whether the d-spacings are reliable (lattice "
                    "confidence >= low). When False, dspacing_fit is "
                    "skipped so the detector-frame artifact d-spacings "
                    "don't inject noise — matches the local-library gate.")


class ExternalMatchOut(BaseModel):
    source: str
    structure_id: str
    formula: str
    space_group: str = ""
    space_group_number: Optional[int] = None
    crystal_system: str = ""
    a_A: Optional[float] = None
    b_A: Optional[float] = None
    c_A: Optional[float] = None
    cif_url: str = ""
    title: str = ""
    # Per-phase fit (0..1, 1=best). Computed when n_fold_scores + d_obs
    # are passed via the external-search request. Null otherwise.
    symmetry_fit: Optional[float] = None
    dspacing_fit: Optional[float] = None
    combined_fit: Optional[float] = None
    # Pattern-degenerate external phases absorbed by this representative
    # (same space group + lattice within 2%). UI: "also matches: …".
    degenerate_with: list[str] = []
    # True if this external phase is structurally already present in the
    # local library (same space group + lattice within 2%). The whole point
    # of external search is finding phases NOT in the library, so these are
    # ranked below novel ones and badged "in library" in the UI.
    in_local_library: bool = False


def _system_from_sg_number(sg_num: Optional[int]) -> str:
    """Crystal system from space group number. Local helper to avoid
    a circular import on the lib service."""
    if sg_num is None:
        return ""
    if 1 <= sg_num <= 2: return "triclinic"
    if 3 <= sg_num <= 15: return "monoclinic"
    if 16 <= sg_num <= 74: return "orthorhombic"
    if 75 <= sg_num <= 142: return "tetragonal"
    if 143 <= sg_num <= 167: return "trigonal"
    if 168 <= sg_num <= 194: return "hexagonal"
    if 195 <= sg_num <= 230: return "cubic"
    return ""


def _cod_result_to_out(r: CodResult) -> ExternalMatchOut:
    return ExternalMatchOut(
        source="COD",
        structure_id=r.cod_id,
        formula=r.formula,
        space_group=r.space_group,
        space_group_number=r.space_group_number,
        crystal_system=_system_from_sg_number(r.space_group_number),
        a_A=r.a_A,
        b_A=r.b_A,
        c_A=r.c_A,
        cif_url=r.cif_url,
        title=r.title,
    )


def _mp_result_to_out(r: MpResult) -> ExternalMatchOut:
    return ExternalMatchOut(
        source="MP",
        structure_id=r.mp_id,
        formula=r.formula,
        space_group=r.space_group,
        space_group_number=r.space_group_number,
        crystal_system=(r.crystal_system
                        or _system_from_sg_number(r.space_group_number)),
        a_A=r.a_A,
        b_A=r.b_A,
        c_A=r.c_A,
        cif_url=r.cif_url,
        title=(
            f"E_above_hull = {r.energy_above_hull_eV:.3f} eV/atom"
            if r.energy_above_hull_eV is not None
            else ""
        ),
    )


@router.post("/external-search", response_model=list[ExternalMatchOut])
async def external_search(req: ExternalSearchRequest):
    """Query external crystallography databases (COD) for candidate phases.

    Run separately from analyze-pixel because the network call is slow
    (~1-5s) and we want analyze-pixel to return immediately with local
    matches. The frontend fires this in parallel after receiving the local
    result.

    Returns [] on network failure (logged) — never raises.

    Chemistry-free mode: when ``elements`` is empty (unknown sample), the
    query falls back to symmetry/lattice only. To avoid pulling the entire
    database, a crystal-system OR lattice bound is REQUIRED in that case —
    otherwise we return [] with a warning rather than a multi-thousand-row
    dump.
    """
    a_range = None
    if req.a_low_A is not None and req.a_high_A is not None:
        a_range = (req.a_low_A, req.a_high_A)

    chemistry_free = not req.elements
    if chemistry_free and req.crystal_system is None and a_range is None:
        # Fail loud rather than silently returning the whole DB or [].
        logger.warning(
            "external-search: chemistry-free query needs a crystal_system "
            "or lattice bound — refusing unconstrained search")
        raise HTTPException(
            status_code=422,
            detail="Chemistry-free external search requires a crystal_system "
                   "or lattice bound (a_low_A/a_high_A) so the database query "
                   "is constrained.",
        )

    # Fire COD and MP queries concurrently. MP is a no-op when no API key
    # is configured (returns []), so we never trip on missing config.
    import asyncio
    cod_task = asyncio.create_task(search_cod_async(
        elements=req.elements,
        system=req.crystal_system,
        a_range_A=a_range,
        max_results=req.max_results,
    ))
    mp_task = asyncio.create_task(search_mp_async(
        elements=req.elements,
        system=req.crystal_system,
        a_range_A=a_range,
        max_results=req.max_results,
    ))
    cod_results, mp_results = await asyncio.gather(
        cod_task, mp_task, return_exceptions=True,
    )

    out: list[ExternalMatchOut] = []
    if isinstance(cod_results, list):
        out.extend(_cod_result_to_out(r) for r in cod_results)
    else:
        logger.warning("COD query raised: %r", cod_results)
    if isinstance(mp_results, list):
        out.extend(_mp_result_to_out(r) for r in mp_results)
    else:
        logger.warning("MP query raised: %r", mp_results)

    # Apply per-phase fits when the frontend passed pattern-derived hints.
    # The same symmetry_fit + dspacing_fit treatment as local matches —
    # phases whose symmetry signature + d-spacings actually fit the pixel
    # rank above phases that just satisfy chemistry + lattice range.
    n_fold_for_fit: Optional[dict[int, float]] = None
    if req.n_fold_scores:
        try:
            n_fold_for_fit = {int(k): float(v) for k, v in req.n_fold_scores.items()}
        except (ValueError, TypeError):
            n_fold_for_fit = None
    # Gate the d-spacing term the same way the local-library path does:
    # only feed observed d-spacings into the fit when they're trustworthy
    # (lattice confidence >= low). Otherwise the detector-frame artifact
    # d-spacings would inject per-candidate noise into the external ranking.
    d_obs_for_fit = (req.d_observed_A or None) if req.d_observed_trustworthy else None

    if n_fold_for_fit or d_obs_for_fit:
        from backend.api.services.crystal_hint_phase_fit import combined_phase_fit
        for m in out:
            s_fit, d_fit, c_fit = combined_phase_fit(
                n_fold_scores=n_fold_for_fit or {},
                crystal_system=m.crystal_system,
                d_observed_A=d_obs_for_fit or [],
                lattice_a_A=m.a_A,
                lattice_c_A=m.c_A,
                space_group=m.space_group,
            )
            m.symmetry_fit = s_fit
            m.dspacing_fit = d_fit
            m.combined_fit = c_fit

    # Flag external phases that are STRUCTURALLY already in the local
    # library (same space group + lattice within 2%, via the same
    # _degenerate test used for collapse). External search exists to find
    # phases the user does NOT already have — surfacing Al / Al2Cu / Si
    # (all local) as the top external hits is noise — so these are pushed
    # below novel phases (not hidden: the user may still want to compare).
    from backend.api.services.crystal_hint_phase_fit import _degenerate
    from backend.api.services.crystal_hint_local_library import all_entries
    _local_keys = [{
        "crystal_system": e.crystal_system, "space_group": e.space_group,
        "sg_number": e.space_group_number,
        "a": e.lattice_a_A, "b": e.lattice_b_A, "c": e.lattice_c_A,
    } for e in all_entries()]
    for m in out:
        m_key = {
            "crystal_system": m.crystal_system, "space_group": m.space_group,
            "sg_number": m.space_group_number,
            "a": m.a_A, "b": m.b_A, "c": m.c_A,
        }
        m.in_local_library = any(_degenerate(m_key, lk) for lk in _local_keys)

    # Sort by relevance. Primary key: novel phases (not already in the local
    # library) first — that's what external search is FOR. Then -combined_fit
    # (higher = better) when fits were computed; otherwise the legacy
    # lattice-distance / E_above_hull / source priority.
    a_target = None
    if a_range is not None:
        a_target = (a_range[0] + a_range[1]) / 2

    def _sort_key(m: ExternalMatchOut) -> tuple:
        a_dist = abs((m.a_A or 0.0) - a_target) if (a_target and m.a_A) else 0.0
        eah = 0.0
        if m.title and "E_above_hull" in m.title:
            try:
                eah = float(m.title.split("=")[1].split("eV")[0].strip())
            except (ValueError, IndexError):
                pass
        src_rank = 0 if m.source == "MP" else 1
        # Primary key = -combined_fit when available (higher fit first).
        # Use 0.0 fallback so unfit matches sort after fit ones — but
        # within the unfit group, the legacy keys still apply.
        fit_key = -(m.combined_fit if m.combined_fit is not None else 0.0)
        # novel-first: already-in-library phases (True→1) sort after novel
        # ones (False→0). This is the PRIMARY key — external search is for
        # discovering phases the user doesn't already have.
        return (m.in_local_library, fit_key, a_dist, eah, src_rank)

    out.sort(key=_sort_key)

    # Collapse pattern-degenerate external candidates the same way local
    # matches are collapsed — COD/MP routinely return several entries with
    # the same space group + lattice (different occupancies/IDs) that an
    # EBSD pattern can't distinguish. The list is already sorted, so the
    # first of each structural group is the representative.
    from backend.api.services.crystal_hint_phase_fit import collapse_degenerate
    _ext_items = [{
        "name": m.formula,
        "crystal_system": m.crystal_system,
        "space_group": m.space_group,
        "sg_number": m.space_group_number,
        "a": m.a_A, "b": m.b_A, "c": m.c_A,
        "_obj": m,
    } for m in out]
    _ext_reps = collapse_degenerate(_ext_items)
    collapsed_out = []
    for rep in _ext_reps:
        obj = rep["_obj"]
        obj.degenerate_with = rep["degenerate_with"]
        collapsed_out.append(obj)
    return collapsed_out


class DownloadCifRequest(BaseModel):
    source: str = Field(..., description="'COD' (others not yet supported)")
    structure_id: str = Field(..., description="ID within the source DB (numeric for COD)")
    save: bool = Field(True, description="Persist to Database/CIF_Library/")
    overwrite: bool = Field(False, description="Replace any existing file with the same name")


class CifMetadataOut(BaseModel):
    formula: str = ""
    space_group: str = ""
    space_group_number: Optional[int] = None
    crystal_system: str = "unknown"
    a_A: Optional[float] = None
    b_A: Optional[float] = None
    c_A: Optional[float] = None
    alpha_deg: Optional[float] = None
    beta_deg: Optional[float] = None
    gamma_deg: Optional[float] = None
    elements: list[str] = []
    n_atoms: int = 0


class DownloadCifResponse(BaseModel):
    success: bool
    source: str
    structure_id: str
    local_path: Optional[str] = None
    metadata: Optional[CifMetadataOut] = None
    warnings: list[str] = []
    error: Optional[str] = None


@router.post("/download-cif", response_model=DownloadCifResponse)
def download_cif_endpoint(req: DownloadCifRequest):
    """Download a CIF from the named external DB, validate it, and write it
    into the local Database/CIF_Library/ directory.

    Does NOT trigger SHT generation — that's a long-running operation that
    must be initiated explicitly from the Simulation page. After the CIF
    is downloaded the user can navigate there and select the new file.
    """
    src = (req.source or "").upper()
    if src == "COD":
        result = download_cif_from_cod(
            cod_id=req.structure_id,
            save=req.save,
            overwrite=req.overwrite,
        )
    elif src == "MP":
        result = download_cif_from_mp(
            mp_id=req.structure_id,
            save=req.save,
            overwrite=req.overwrite,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source '{req.source}'. Currently supported: COD, MP.",
        )

    meta_out = None
    if result.metadata is not None:
        m = result.metadata
        meta_out = CifMetadataOut(
            formula=m.formula,
            space_group=m.space_group,
            space_group_number=m.space_group_number,
            crystal_system=m.crystal_system,
            a_A=m.a_A, b_A=m.b_A, c_A=m.c_A,
            alpha_deg=m.alpha_deg, beta_deg=m.beta_deg, gamma_deg=m.gamma_deg,
            elements=list(m.elements),
            n_atoms=m.n_atoms,
        )

    return DownloadCifResponse(
        success=result.success,
        source=result.source,
        structure_id=result.structure_id,
        local_path=str(result.local_path) if result.local_path else None,
        metadata=meta_out,
        warnings=result.warnings,
        error=result.error,
    )


@router.get("/library")
def list_library():
    """Return the local CIF/SHT library index (UI debug + reference)."""
    entries = get_index()
    return [
        {
            "key": e.key,
            "formula": e.formula,
            "display_formula": e.display_formula or e.formula,
            "space_group": e.space_group,
            "space_group_number": e.space_group_number,
            "crystal_system": e.crystal_system,
            "elements": list(e.elements),
            "lattice_a_A": e.lattice_a_A,
            "lattice_b_A": e.lattice_b_A,
            "lattice_c_A": e.lattice_c_A,
            "n_atoms": e.n_atoms,
            "has_cif": e.cif_path is not None,
            "has_xtal": e.xtal_path is not None,
            "has_sht": e.sht_path is not None,
            "parse_error": e.parse_error,
        }
        for e in entries.values()
    ]
