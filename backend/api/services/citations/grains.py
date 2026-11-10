"""The citable step of a grain reconstruction, built from the settings it was made with.

The grains of the Analysis page (and of each batch file) are not an indexing result: there is no
provenance trail to record them on (see the note on ``analysis.grain_reconstruction`` in ``steps.py``).
``grain_reconstruction_step`` therefore turns the reconstruction parameters and the grain-size
statistics settings into the ``{"key", "params"}`` entry that ``citation_bundle`` and ``render_methods``
take, with the prose the sentence needs already chosen here, where the meaning of each setting is known.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from backend.api.services.app_version import get_version_info
from backend.api.services.citations.provenance import _jsonable

GRAIN_STEP_KEY = "analysis.grain_reconstruction"

# What each edge-grain rule leaves out of the grain-size statistics (analysis.grain_analysis.find_edge_grains).
_EDGE_RULE_GRAINS = {
    "matlab_script": "grains with a measured pixel in the outermost two pixel rows or columns of the measured area",
    "mtex": "grains touching the map border, or a not-indexed area open to the border,",
    "outer_row": "grains with a pixel in the first or last row or column of the map",
}
# When the rule keeps every grain instead (analysis.grain_analysis.edge_safeguard_triggered).
_EDGE_SAFEGUARD = {
    "matlab_script": "fewer than 50 grains, or less than 15 % of the grains, would have remained",
    "mtex": "fewer than 50 grains, or less than 15 % of the grains, would have remained",
    "outer_row": "more than 85 % of the grains would have been left out",
}
_ECD_BASIS = {
    "cell_area": "the area of the grain's measurement cells, covered not-indexed pixels included",
    "measured_pixels": "the number of measured pixels of the grain times the square of the step size",
}


def _edge_rule_text(edge_rule: str, grain_size: Any, one_phase: bool) -> str:
    """The edge rule as a clause, with what it did when the statistics (``grain_size``) are known.

    ``one_phase``: the statistics are of one phase. The script's rule then draws its strip around that phase's
    own measured pixels (``find_edge_grains``), which the clause says."""
    if edge_rule == "keep_all":
        return "no grain was left out of them for touching the map border"
    if edge_rule not in _EDGE_RULE_GRAINS:
        return f"the edge-grain rule was '{edge_rule}'"
    who, why = _EDGE_RULE_GRAINS[edge_rule], _EDGE_SAFEGUARD[edge_rule]
    if edge_rule == "matlab_script" and one_phase:
        who += " of that phase"
    if grain_size is None:
        return f"{who} were left out of them, unless {why}"
    n_total = int(getattr(grain_size, "n_grains_total", 0))
    n_edge = int(getattr(grain_size, "n_edge_grains", 0))
    if bool(getattr(grain_size, "edge_safeguard_applied", False)):
        return (f"{who} would have been left out of them, but all {n_total} grains were kept, "
                f"because {why}")
    if n_total == 0:
        return f"{who} would have been left out of them, but there were no grains to count"
    return f"{who} were left out of them ({n_edge} of {n_total} grains)"


def _stats_phase_text(grain_size: Any, stats_phase: Any) -> str:
    if grain_size is not None:
        if getattr(grain_size, "stats_phase", None) is None:
            return "the grains of all phases"
        return f"the grains of the phase {getattr(grain_size, 'stats_phase_name', None) or grain_size.stats_phase}"
    if stats_phase is None:
        return "the grains of all phases"
    if stats_phase == "most_measured":
        return "the grains of the phase with the most measured pixels"
    return f"the grains of phase {stats_phase}"


def grain_reconstruction_step(recon: Optional[Mapping[str, Any]], *, edge_rule: str, ecd_basis: str,
                              stats_phase: Any = None, grain_size: Any = None) -> dict:
    """``{"key": "analysis.grain_reconstruction", "params": {...}}`` for one reconstruction.

    ``recon``: the reconstruction parameters as the Analysis route keeps them (``gb_threshold_deg``,
    ``min_pixels``, ``alpha``, ``fill``); a missing one is rendered "[... not recorded]", never guessed.
    ``edge_rule``, ``ecd_basis``, ``stats_phase``: the grain-size statistics settings. ``grain_size``: their
    result (``analysis.grain_analysis.GrainSizeResult``) when it was computed; it adds the number of edge
    grains, whether the safeguard kept them all, and the name of the statistics phase.
    """
    recon = dict(recon or {})
    alpha = recon.get("alpha")
    min_pixels = recon.get("min_pixels")
    one_phase = (getattr(grain_size, "stats_phase", None) is not None if grain_size is not None
                 else stats_phase is not None)
    params = {
        "threshold_deg": recon.get("gb_threshold_deg"),
        "min_pixels": min_pixels,
        # only a minimum above 1 culls anything and segments a second time (grain_reconstruction.segment).
        # Unknown: the method's clause is kept and names the missing number ("[min_pixels not recorded]").
        "culls": True if min_pixels is None else bool(int(min_pixels) > 1),
        "alpha": alpha,
        # the closing disc has radius floor(alpha) (grain_reconstruction._disc): below 1 it closes nothing.
        # Unknown: as for culls, the clause names the missing alpha.
        "closes_gaps": True if alpha is None else bool(float(alpha) >= 1),
        "fill": None if recon.get("fill") is None else bool(recon.get("fill")),
        "edge_rule": edge_rule,
        "ecd_basis": ecd_basis,
        "stats_phase": stats_phase,
        "edge_rule_text": _edge_rule_text(edge_rule, grain_size, one_phase),
        "ecd_basis_text": _ECD_BASIS.get(ecd_basis, f"'{ecd_basis}'"),
        "stats_phase_text": _stats_phase_text(grain_size, stats_phase),
        "orienta_version": get_version_info().get("version"),
    }
    return {"key": GRAIN_STEP_KEY, "params": _jsonable(params)}
