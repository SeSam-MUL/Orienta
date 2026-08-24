"""Real-data acceptance for the EDS chemistry phase map.

Baselines measured 2026-08-19 on SampleB (see the design spec, section 2):
the old mean-deviation rule put 20.95 % of all pixels on an Fe-bearing phase
they could not be, and left the Si particles unclassified. Skips when the
test data is absent so a clean checkout still passes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

H5 = ROOT / "Test_data" / (
    "EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 "
    "Elementverteilungsdaten 1.h5oina"
)
DB = ROOT / "Database" / "crystal_database.xlsx"

pytestmark = pytest.mark.skipif(
    not (H5.is_file() and DB.is_file()),
    reason="SampleB test data or crystal_database.xlsx not available",
)


@pytest.fixture(scope="module")
def sampleb():
    """(at% maps, n_rows, n_cols, candidate entries) for SampleB."""
    import h5py
    from eds_utils import (
        parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct,
    )
    from tools.h5_viewer_backend import H5OINADataExtractor
    from backend.api.services.cif_phase_library import load_cif_phase_library

    with h5py.File(str(H5), "r") as f:
        ext = H5OINADataExtractor(f, "Oxford")
        n_rows, n_cols = ext.get_grid_dimensions()
        counts = {}
        for name in ext.get_available_elements():
            d = ext.get_element_map(name)
            if d is not None:
                counts[parse_element_name(name)] = d.astype(np.float64)
    at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts))
    return at, n_rows, n_cols, load_cif_phase_library(DB)


@pytest.fixture(scope="module")
def classified(sampleb):
    from backend.api.services.cif_phase_library import auto_classify_pixels
    at, n_rows, n_cols, lib = sampleb
    grid, score, cands, ambiguous = auto_classify_pixels(
        at_pct_per_element=at, n_rows=n_rows, n_cols=n_cols, cif_library=lib,
    )
    return at, grid.ravel(), score.ravel(), cands, ambiguous.ravel()


def _labels(grid, cands):
    names = np.array(["UNCLASSIFIED"] + [c.cif_filename for c in cands])
    return names[grid + 1]


def test_fe_phases_are_not_assigned_to_iron_free_pixels(classified):
    """Was 20.95 % with the old rule, 12.59 % with plain chemistry_fit."""
    at, grid, _score, cands, _amb = classified
    fe = np.asarray(at["Fe"], dtype=float)
    bad = 0
    for idx, entry in enumerate(cands):
        need = entry.composition.get("Fe", 0.0)
        if need < 5.0:
            continue
        m = (grid == idx)
        bad += int((m & (fe < 0.25 * need)).sum())
    frac = bad / grid.size * 100
    assert frac < 3.0, (
        f"{frac:.2f}% of pixels carry an Fe-phase with <25% of its required "
        f"Fe (old rule 20.95%, plain chemistry_fit 12.59%, target <3%)"
    )


def test_silicon_particles_are_silicon(classified):
    """Si-dominated pixels must be Si, not an aluminide."""
    at, grid, _score, cands, _amb = classified
    si = np.asarray(at["Si"], dtype=float)
    labels = _labels(grid, cands)
    particles = si > 60.0
    assert particles.sum() >= 15, "expected ~20 Si-dominated pixels in SampleB"
    hit = (labels[particles] == "Si.cif").mean()
    assert hit > 0.9, f"only {hit*100:.0f}% of Si particles labelled Si.cif"


def test_no_silicon_particle_gets_an_iron_phase(classified):
    """The old rule put Mn2(AlSi)5 and Al3FeSi2 on Si particles."""
    at, grid, _score, cands, _amb = classified
    si = np.asarray(at["Si"], dtype=float)
    fe_phases = {i for i, e in enumerate(cands)
                 if e.composition.get("Fe", 0.0) >= 5.0}
    offenders = int(sum(1 for g in grid[si > 40.0] if int(g) in fe_phases))
    assert offenders == 0, f"{offenders} Si-rich pixels assigned an Fe phase"


def test_the_real_particle_survives_the_veto(classified):
    """The relative veto must not eat the particle it is meant to protect.

    Ground truth is the connected Fe-rich region (Fe > 4 at%, 3401 px on
    SampleB). It measures a mean 5.33 at% Fe against the 11.6 at% its phase
    nominally requires — a 0.46x standardless Cliff-Lorimer under-read — so
    a veto threshold anywhere near 0.46 silently deletes it. Measured:
    rel_req 0.30 keeps 99.4 % of it, 0.40 keeps only 71.5 %.

    This is the assertion the first calibration lacked; it was chosen on
    per-pixel correctness alone and picked 0.40.
    """
    at, grid, _score, cands, _amb = classified
    fe = np.asarray(at["Fe"], dtype=float)
    particle = fe > 4.0
    assert particle.sum() > 2000, "expected a substantial Fe-rich particle"
    fe_phases = [i for i, e in enumerate(cands)
                 if e.composition.get("Fe", 0.0) >= 5.0]
    kept = np.isin(grid, fe_phases)[particle].mean()
    assert kept > 0.95, (
        f"only {kept*100:.1f}% of the real particle kept an Fe-bearing phase "
        f"— the relative veto is too aggressive for this quantification"
    )


def test_clean_matrix_is_aluminium(classified):
    at, grid, _score, cands, _amb = classified
    al = np.asarray(at["Al"], dtype=float)
    labels = _labels(grid, cands)
    matrix = al >= 95.0
    assert matrix.sum() > 100, "expected a substantial clean-matrix population"
    hit = (labels[matrix] == "Al.cif").mean()
    assert hit > 0.9, f"only {hit*100:.0f}% of clean matrix labelled Al.cif"


def test_ambiguity_grid_is_returned_and_plausible(classified):
    _at, grid, _score, _cands, amb = classified
    assert amb.shape == grid.shape
    assert amb.dtype == bool
    # Unclassified pixels are never flagged ambiguous.
    assert not amb[grid < 0].any()


# ---------------------------------------------------------------------------
# Cluster mode — the SHIPPED DEFAULT. A review found the acceptance suite
# only ever exercised the per-pixel path, so the mode the user actually gets
# was unmeasured on real data.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clustered(sampleb):
    from backend.api.services.eds_clustering import cluster_and_match
    at, n_rows, n_cols, lib = sampleb
    measured = set(at)
    cands = [e for e in lib.values() if set(e.elements).issubset(measured)]
    grid, cgrid, matches, k = cluster_and_match(at, n_rows, n_cols, cands)
    return at, grid.ravel(), cgrid.ravel(), matches, k, cands


def test_cluster_mode_clean_matrix_is_aluminium(clustered):
    at, grid, _cg, _m, _k, cands = clustered
    al = np.asarray(at["Al"], dtype=float)
    labels = _labels(grid, cands)
    matrix = al >= 95.0
    hit = (labels[matrix] == "Al.cif").mean()
    assert hit > 0.9, f"cluster mode: only {hit*100:.0f}% of clean matrix is Al.cif"


def test_cluster_mode_silicon_particles_are_silicon(clustered):
    at, grid, _cg, _m, _k, cands = clustered
    si = np.asarray(at["Si"], dtype=float)
    labels = _labels(grid, cands)
    hit = (labels[si > 60.0] == "Si.cif").mean()
    assert hit > 0.9, f"cluster mode: only {hit*100:.0f}% of Si particles are Si.cif"


def test_cluster_mode_keeps_the_real_particle(clustered):
    at, grid, _cg, _m, _k, cands = clustered
    fe = np.asarray(at["Fe"], dtype=float)
    particle = fe > 4.0
    fe_phases = [i for i, e in enumerate(cands)
                 if e.composition.get("Fe", 0.0) >= 5.0]
    kept = np.isin(grid, fe_phases)[particle].mean()
    assert kept > 0.95, f"cluster mode kept only {kept*100:.1f}% of the particle"


def test_cluster_mode_is_more_spatially_coherent_than_per_pixel(
        classified, clustered):
    """The reason cluster mode exists: grain-like regions, not confetti.

    Measured on SampleB: cluster ~0.96, per-pixel ~0.69.
    """
    from backend.api.services.eds_clustering import phase_coherence
    _at, pix_grid, _s, _c, _a = classified
    at, cl_grid, _cg, _m, _k, _cands = clustered
    n_rows, n_cols = 90, 120
    coh_pix = phase_coherence(pix_grid.reshape(n_rows, n_cols))
    coh_cl = phase_coherence(cl_grid.reshape(n_rows, n_cols))
    assert coh_cl > coh_pix, f"cluster {coh_cl:.3f} !> pixel {coh_pix:.3f}"
    assert coh_cl > 0.9


def test_cluster_mode_agrees_with_band_contrast(clustered):
    """Non-circular check: Band Contrast comes from the diffraction
    patterns, not from chemistry. If the chemistry-derived second phase is
    real, it must sit where band contrast drops. Measured: r = -0.84.

    Restricted to measured pixels — an unmeasured region has BC 0 AND EDS 0
    and would manufacture this correlation out of nothing.
    """
    import h5py
    from backend.api.services.chemistry_score import has_chemistry
    from tools.h5_viewer_backend import H5OINADataExtractor

    at, grid, _cg, _m, _k, cands = clustered
    with h5py.File(str(H5), "r") as f:
        bc = np.asarray(
            H5OINADataExtractor(f, "Oxford").get_band_contrast_map(), dtype=float
        ).ravel()

    live = has_chemistry(at)
    fe_phases = [i for i, e in enumerate(cands)
                 if e.composition.get("Fe", 0.0) >= 5.0]
    is_fe = np.isin(grid, fe_phases).astype(float)
    r = float(np.corrcoef(is_fe[live], bc[live])[0, 1])
    assert r < -0.5, (
        f"the Fe-bearing region does not track band contrast (r={r:+.3f}) — "
        f"the chemistry map disagrees with the independent structural evidence"
    )


def test_unmeasured_pixels_are_never_assigned_a_phase(sampleb):
    """A pixel with no EDS counts must come out unclassified in BOTH modes.

    Regression for: chemistry_fit's neutral 1.0 for "no chemistry" is the
    TOP of the score range, so every candidate tied at the maximum and
    argmax handed the pixel to library row 1 at a perfect score. Measured on
    a real 7050 scan: 63.7 % of the map fabricated this way.
    """
    from backend.api.services.cif_phase_library import auto_classify_pixels
    from backend.api.services.eds_clustering import cluster_and_match

    at, n_rows, n_cols, lib = sampleb
    # Blank out a corner so the file has a genuinely unmeasured region.
    holed = {el: np.array(v, dtype=float).copy() for el, v in at.items()}
    dead = np.zeros(n_rows * n_cols, dtype=bool)
    dead[: 10 * n_cols] = True
    for v in holed.values():
        v[dead] = 0.0

    grid, _s, cands, _a = auto_classify_pixels(holed, n_rows, n_cols, lib)
    assert (grid.ravel()[dead] == -1).all(), "pixel mode assigned a dead region"

    measured = set(holed)
    cl_cands = [e for e in lib.values() if set(e.elements).issubset(measured)]
    pgrid, _cg, _m, _k = cluster_and_match(holed, n_rows, n_cols, cl_cands)
    assert (pgrid.ravel()[dead] == -1).all(), "cluster mode assigned a dead region"


# ---------------------------------------------------------------------------
# Ratio matching — the case the relative veto made impossible.
# Spec: docs/superpowers/specs/2026-08-20-eds-ratio-matching-design.md
# ---------------------------------------------------------------------------

CU_FILE = ROOT / "Test_data" / "batch_test" / (
    "7050EBSD 70502_R Arbeitsbereich 1 Elementverteilungsdaten 5.h5oina"
)


@pytest.mark.skipif(not (CU_FILE.is_file() and DB.is_file()),
                    reason="7050 Cu test file not available")
def test_a_copper_phase_is_assignable_again():
    """`Al7FeCu2` scored 0.050 — the veto floor — on EVERY pixel of both
    Cu-bearing test files under the relative-to-nominal veto. It needs
    20 at% Cu nominally; the veto demanded 6 at% measured; a thin ring seen
    through a large interaction volume never reads that.

    Here it is scored on a real pixel where Cu and Fe are both enriched and
    sit at roughly the nominal Cu/Fe = 2, while the absolute values are
    about 6x below stoichiometry. That is the whole point of matching on
    ratios: the dilution divides out.
    """
    import h5py
    from eds_utils import (
        parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct,
    )
    from tools.h5_viewer_backend import H5OINADataExtractor
    from backend.api.services.chemistry_score import (
        background_levels, infer_matrix_element, score_phase_ratio,
    )

    with h5py.File(str(CU_FILE), "r") as f:
        ext = H5OINADataExtractor(f, "Oxford")
        counts = {}
        for name in ext.get_available_elements():
            d = ext.get_element_map(name)
            if d is not None:
                counts[parse_element_name(name)] = d.astype(np.float64)
    at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts))

    fe = np.asarray(at["Fe"], dtype=float)
    cu = np.asarray(at["Cu"], dtype=float)
    both = (fe > np.percentile(fe, 95)) & (cu > np.percentile(cu, 95))
    assert both.sum() > 10, "expected a Cu+Fe enriched population"

    al7fecu2 = {"Al": 70.0, "Fe": 10.0, "Cu": 20.0}
    matrix = infer_matrix_element(at)
    bg = background_levels(at)
    scores = [
        float(score_phase_ratio(
            {el: np.array([float(np.asarray(v)[j])]) for el, v in at.items()},
            al7fecu2, matrix_element=matrix, no_data_score=0.0, background=bg)[0])
        for j in np.where(both)[0]
    ]
    assert max(scores) > 0.5, (
        f"best Al7FeCu2 score is {max(scores):.3f} — the Cu phase is still "
        f"unassignable (it was pinned at the 0.050 veto floor before)"
    )


@pytest.mark.skipif(not (H5.is_file() and DB.is_file()),
                    reason="SampleB test data not available")
def test_the_map_and_the_suggestion_panel_agree():
    """They used to disagree on the same pixel: the panel ran a
    tolerance-based scorer and the map ran the vetoed one, so a Cu-rich
    pixel read 0.415 in the panel and 0.050 on the map."""
    import h5py
    from eds_utils import (
        parse_element_name, counts_to_weight_pct, weight_pct_to_atomic_pct,
    )
    from tools.h5_viewer_backend import H5OINADataExtractor
    from backend.api.services.cif_phase_library import (
        candidates_for, load_cif_phase_library, suggest_phases_from_cif_library,
    )
    from backend.api.services.chemistry_score import (
        background_levels, infer_matrix_element, score_phase_ratio,
    )

    with h5py.File(str(H5), "r") as f:
        ext = H5OINADataExtractor(f, "Oxford")
        counts = {}
        for name in ext.get_available_elements():
            d = ext.get_element_map(name)
            if d is not None:
                counts[parse_element_name(name)] = d.astype(np.float64)
    at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts))
    lib = load_cif_phase_library(DB)
    matrix = infer_matrix_element(at)
    bg = background_levels(at)

    fe = np.asarray(at["Fe"], dtype=float)
    for j in (int(np.argmax(fe)), int(np.argmin(fe))):
        pix = {el: np.array([float(np.asarray(v)[j])]) for el, v in at.items()}
        cands = candidates_for(lib, at.keys())
        map_best = max(cands, key=lambda e: float(score_phase_ratio(
            pix, e.composition, matrix_element=matrix,
            no_data_score=0.0, background=bg)[0]))
        panel = suggest_phases_from_cif_library(
            {el: float(v[0]) for el, v in pix.items()}, lib,
            min_score=0.0, matrix_element=matrix, background=bg)
        assert panel, "the panel returned nothing"
        assert panel[0]["cif_filename"] == map_best.cif_filename, (
            f"panel says {panel[0]['cif_filename']}, "
            f"map says {map_best.cif_filename}"
        )
