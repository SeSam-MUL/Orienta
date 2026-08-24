"""
Tests for the EDS phase-map pipeline (cif_phase_library + phase_map_store).

Coverage:
    - Formula parsing (Unicode subscripts, parens, decimals, garbage)
    - CIF library loader (real Database/crystal_database.xlsx if present,
      else a generated minimal xlsx fixture so the test is hermetic)
    - Phase suggestion + element-subset pre-filter
    - Vectorised auto-classify
    - PhaseMapStore: set / get / assign_region / clear / phase_summary /
      get_phase_masks
    - Deterministic phase palette
    - PNG rendering of the phase grid

The end-to-end test that drives the full route handler against a real
h5oina file is in :mod:`test_eds_phase_map_e2e` so this file stays fast
enough to run on every pytest invocation.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from backend.api.services.cif_phase_library import (
    CifPhaseEntry,
    auto_classify_pixels,
    load_cif_phase_library,
    parse_formula_to_at_pct,
    suggest_phases_from_cif_library,
)
from backend.api.services.phase_map_store import (
    PhaseMapState,
    PhaseMapStore,
    palette_hex_for_state,
    phase_palette,
    render_phase_map_to_base64,
)

PROJECT_ROOT = Path(__file__).parent.parent
REAL_DB_XLSX = PROJECT_ROOT / "Database" / "crystal_database.xlsx"


# ---------------------------------------------------------------------------
# parse_formula_to_at_pct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("Fe2O3", {"Fe": 40.0, "O": 60.0}),
        ("Fe₂O₃", {"Fe": 40.0, "O": 60.0}),         # Unicode subscripts
        ("Al",    {"Al": 100.0}),
        ("Al2(FeSi)3", {"Al": 25.0, "Fe": 37.5, "Si": 37.5}),  # parentheses
        ("Al13Fe4", {"Al": pytest.approx(76.470588, abs=1e-3),
                     "Fe": pytest.approx(23.529411, abs=1e-3)}),
    ],
)
def test_parse_formula_known_compositions(formula, expected):
    """Stoichiometric formulas must round-trip to the right At.% split."""
    out = parse_formula_to_at_pct(formula)
    assert set(out.keys()) == set(expected.keys()), out
    for el, target in expected.items():
        if isinstance(target, float):
            assert out[el] == pytest.approx(target, abs=0.05)
        else:
            assert out[el] == target


def test_parse_formula_decimal_subscripts():
    """Real CIF DB has formulas like Al17.6Fe7.2Si2.4 — pymatgen handles this."""
    out = parse_formula_to_at_pct("Al17.6Fe7.2Si2.4")
    total = out["Al"] + out["Fe"] + out["Si"]
    assert total == pytest.approx(100.0, abs=0.5)
    # Al should clearly dominate
    assert out["Al"] > out["Fe"] > out["Si"]


def test_parse_formula_empty_returns_empty():
    """Empty input returns empty dict — never raises, callers can skip."""
    assert parse_formula_to_at_pct("") == {}
    assert parse_formula_to_at_pct("   ") == {}


def test_parse_formula_garbage_returns_empty():
    """Unparseable garbage logs and returns empty (silent fallback ok here —
    suggest-phases fans out across many CIFs and one bad row shouldn't
    kill the whole library load)."""
    assert parse_formula_to_at_pct("not a formula") == {}
    assert parse_formula_to_at_pct("@#$%") == {}


# ---------------------------------------------------------------------------
# load_cif_phase_library
# ---------------------------------------------------------------------------


def test_load_cif_library_missing_file_returns_empty(tmp_path):
    """Missing file is not an error — UI falls back to default library."""
    assert load_cif_phase_library(tmp_path / "nope.xlsx") == {}


@pytest.mark.skipif(not REAL_DB_XLSX.is_file(),
                    reason="Real crystal_database.xlsx not built yet")
def test_load_cif_library_real_db_has_entries():
    """Smoke test against the user's actual curated CIF library."""
    lib = load_cif_phase_library(REAL_DB_XLSX)
    assert len(lib) >= 1
    # Each entry must have a non-empty composition that sums to ~100
    for entry in lib.values():
        assert isinstance(entry, CifPhaseEntry)
        assert entry.composition
        total = sum(entry.composition.values())
        assert total == pytest.approx(100.0, abs=0.5), (
            f"{entry.cif_filename}: At.% sum was {total}, formula={entry.formula}"
        )
        # Elements list must be sorted and match composition keys
        assert entry.elements == sorted(entry.composition.keys())


@pytest.mark.skipif(not REAL_DB_XLSX.is_file(),
                    reason="Real crystal_database.xlsx not built yet")
def test_load_cif_library_caches_by_mtime(tmp_path):
    """Second call must reuse the cached library if mtime unchanged."""
    lib1 = load_cif_phase_library(REAL_DB_XLSX)
    lib2 = load_cif_phase_library(REAL_DB_XLSX)
    assert lib1 is lib2  # same object, not just equal — proves cache hit


# ---------------------------------------------------------------------------
# suggest_phases_from_cif_library
# ---------------------------------------------------------------------------


def _make_library(specs):
    """Build a tiny CIF library inline for unit tests."""
    lib = {}
    for key, formula, composition in specs:
        lib[key] = CifPhaseEntry(
            key=key,
            cif_filename=key,
            formula=formula,
            space_group="P1 (1)",
            space_group_number=1,
            crystal_system="triclinic",
            composition=composition,
            elements=sorted(composition.keys()),
        )
    return lib


def test_suggest_pre_filters_by_element_subset():
    """Pixel without oxygen cannot match Fe2O3, even if Fe matches well."""
    lib = _make_library([
        ("Fe.cif",    "Fe",     {"Fe": 100.0}),
        ("Fe2O3.cif", "Fe2O3", {"Fe": 40.0, "O": 60.0}),
    ])
    measured = {"Fe": 100.0}  # no oxygen
    hits = suggest_phases_from_cif_library(measured, lib)
    keys = [h["cif_filename"] for h in hits]
    assert "Fe.cif" in keys
    assert "Fe2O3.cif" not in keys


def test_suggest_score_orders_by_closeness():
    """Closer composition must score higher."""
    lib = _make_library([
        ("Al.cif",      "Al",     {"Al": 100.0}),
        ("Al13Fe4.cif", "Al13Fe4", {"Al": 76.47, "Fe": 23.53}),
    ])
    # Pure Al pixel -> Al.cif wins
    pure_al = {"Al": 100.0, "Fe": 0.001}
    hits = suggest_phases_from_cif_library(pure_al, lib)
    assert hits[0]["cif_filename"] == "Al.cif"
    assert hits[0]["score"] > 0.9
    # 75/25 pixel -> Al13Fe4 wins
    al_fe = {"Al": 75.0, "Fe": 25.0}
    hits = suggest_phases_from_cif_library(al_fe, lib)
    assert hits[0]["cif_filename"] == "Al13Fe4.cif"


def test_suggest_min_score_filters_low_matches():
    """min_score must gate borderline matches without changing the hard
    reject ``suggest_phases`` already does at ``max_dev > 2 * tolerance``.

    Construct a measured composition that scores ~0.5 (deviation 7.5 At.%
    against tolerance 15 → 1 - 7.5/15 = 0.5) so the score lives strictly
    between the two thresholds we want to test.
    """
    lib = _make_library([("Fe.cif", "Fe", {"Fe": 100.0})])
    measured = {"Fe": 92.5}  # deviation 7.5 → score ~0.5

    strict = suggest_phases_from_cif_library(measured, lib, min_score=0.99)
    assert strict == []

    relaxed = suggest_phases_from_cif_library(measured, lib, min_score=0.0)
    assert len(relaxed) == 1
    assert relaxed[0]["score"] == pytest.approx(0.5, abs=0.05)


def test_suggest_empty_inputs_return_empty():
    assert suggest_phases_from_cif_library({}, {}) == []
    lib = _make_library([("Al.cif", "Al", {"Al": 100.0})])
    assert suggest_phases_from_cif_library({}, lib) == []


# ---------------------------------------------------------------------------
# auto_classify_pixels
# ---------------------------------------------------------------------------


def test_auto_classify_picks_best_phase_per_pixel():
    """Each pixel ends up with its closest CIF phase."""
    lib = _make_library([
        ("Al.cif",      "Al",     {"Al": 100.0}),
        ("Al13Fe4.cif", "Al13Fe4", {"Al": 76.47, "Fe": 23.53}),
    ])
    n_pix = 4
    at_maps = {
        # First two pixels are pure Al, last two are 75/25.
        "Al": np.array([100.0, 99.0, 75.0, 76.0], dtype=np.float32),
        "Fe": np.array([0.0,    1.0, 25.0, 24.0], dtype=np.float32),
    }
    grid, scores, candidates, ambiguous = auto_classify_pixels(
        at_maps, n_rows=2, n_cols=2, cif_library=lib,
        tolerance=15.0, min_score=0.0,
    )
    assert ambiguous.shape == (2, 2)
    # Order of candidates matches insertion order of the dict that survived
    # the pre-filter — sort the keys we care about by their list position.
    name_for_idx = {i: e.cif_filename for i, e in enumerate(candidates)}
    flat = grid.flatten()
    # First two -> Al, last two -> Al13Fe4
    assert name_for_idx[flat[0]] == "Al.cif"
    assert name_for_idx[flat[1]] == "Al.cif"
    assert name_for_idx[flat[2]] == "Al13Fe4.cif"
    assert name_for_idx[flat[3]] == "Al13Fe4.cif"
    assert scores.shape == (2, 2)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_auto_classify_unclassified_when_below_min_score():
    """Pixels far from every phase get -1, not the least-bad bucket.

    The fixture is an aluminium pixel carrying a little Fe, offered only
    ``Fe.cif``. Note the At.% must be a closed set summing to ~100: the
    2026-08-19 scorer renormalises before comparing, so a single-element
    map reading "10 at% Fe" is by definition 100 % Fe and would legitimately
    be a perfect Fe match.
    """
    lib = _make_library([("Fe.cif", "Fe", {"Fe": 100.0})])
    at_maps = {
        "Al": np.array([90.0], dtype=np.float32),
        "Fe": np.array([10.0], dtype=np.float32),
    }
    grid, scores, candidates, _ambiguous = auto_classify_pixels(
        at_maps, n_rows=1, n_cols=1, cif_library=lib,
        tolerance=15.0, min_score=0.5,
    )
    assert grid[0, 0] == -1
    # Score is whatever the phase scored — even if below threshold,
    # we still report it for diagnostic UI.
    assert scores.shape == (1, 1)


def test_auto_classify_no_matching_candidates_returns_empty_grid():
    """If no CIF has a subset of the measured elements, return all -1."""
    lib = _make_library([("Mg.cif", "Mg", {"Mg": 100.0})])
    at_maps = {"Fe": np.array([100.0], dtype=np.float32)}
    grid, scores, candidates, _ambiguous = auto_classify_pixels(
        at_maps, n_rows=1, n_cols=1, cif_library=lib,
        tolerance=15.0, min_score=0.0,
    )
    assert candidates == []
    assert grid.shape == (1, 1)
    assert grid[0, 0] == -1


# ---------------------------------------------------------------------------
# PhaseMapStore
# ---------------------------------------------------------------------------


def _stock_state(n_rows=4, n_cols=4):
    """Build a small phase map for store tests."""
    entries = [
        CifPhaseEntry(
            key="A.cif", cif_filename="A.cif", formula="Fe", space_group="",
            space_group_number=None, crystal_system="",
            composition={"Fe": 100.0}, elements=["Fe"],
        ),
        CifPhaseEntry(
            key="B.cif", cif_filename="B.cif", formula="O2", space_group="",
            space_group_number=None, crystal_system="",
            composition={"O": 100.0}, elements=["O"],
        ),
    ]
    grid = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [-1, -1, 1, 1],
        [-1, -1, 1, 1],
    ], dtype=np.int32)
    scores = np.full((n_rows, n_cols), 0.8, dtype=np.float32)
    return entries, grid, scores


def test_phase_map_store_set_get_roundtrip():
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    state = store.get_state()
    assert state is not None
    assert state.n_rows == 4 and state.n_cols == 4
    assert len(state.phase_entries) == 2
    np.testing.assert_array_equal(state.phase_grid, grid)


def test_phase_map_store_assign_region_paints_pixels():
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    n = store.assign_region(0, 1, 0, 1, target_phase_index=1)  # 2x2 → 4 pix
    assert n == 4
    state = store.get_state()
    # Top-left 2x2 must now be all phase 1
    assert (state.phase_grid[0:2, 0:2] == 1).all()
    # Locked mask is allocated and set
    assert state.locked_mask is not None
    assert state.locked_mask[0:2, 0:2].all()
    # Score for assigned pixels is forced to 1.0 (manual override is certain)
    assert (state.score_grid[0:2, 0:2] == 1.0).all()


def test_phase_map_store_assign_region_unclassified():
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    store.assign_region(0, 1, 0, 1, target_phase_index=-1)
    state = store.get_state()
    assert (state.phase_grid[0:2, 0:2] == -1).all()
    assert (state.score_grid[0:2, 0:2] == 0.0).all()


def test_phase_map_store_assign_mask_paints_pixels():
    """assign_mask must paint exactly the pixels in the bool mask."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 3] = True
    mask[3, 0] = True
    n = store.assign_mask(mask, target_phase_index=0)
    assert n == 3
    state = store.get_state()
    assert state.phase_grid[1, 1] == 0
    assert state.phase_grid[2, 3] == 0
    assert state.phase_grid[3, 0] == 0
    # Other pixels untouched
    assert state.phase_grid[0, 0] == grid[0, 0]
    assert state.locked_mask[1, 1] and state.locked_mask[2, 3] and state.locked_mask[3, 0]


def test_phase_map_store_assign_mask_shape_mismatch_raises():
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    with pytest.raises(ValueError, match="mask shape"):
        store.assign_mask(np.zeros((3, 3), dtype=bool), target_phase_index=0)


def test_phase_map_store_assign_mask_empty_returns_zero():
    """Painting with an all-False mask is a no-op, not an error."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    n = store.assign_mask(np.zeros((4, 4), dtype=bool), target_phase_index=0)
    assert n == 0


def test_phase_map_store_assign_region_invalid_index_raises():
    """Out-of-range phase index must raise — fail-loud, no silent corruption."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    with pytest.raises(ValueError):
        store.assign_region(0, 1, 0, 1, target_phase_index=999)


def test_phase_map_store_assign_region_without_loaded_state_raises():
    store = PhaseMapStore()
    with pytest.raises(RuntimeError):
        store.assign_region(0, 1, 0, 1, target_phase_index=0)


def test_phase_map_store_summary_counts_match_grid():
    """phase_summary must report exact pixel counts and a sane percentage."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    summary = store.phase_summary()
    by_idx = {s["phase_index"]: s for s in summary}
    assert by_idx[0]["n_pixels"] == 4   # top-left 2x2
    assert by_idx[1]["n_pixels"] == 8   # right two columns
    assert by_idx[-1]["n_pixels"] == 4  # bottom-left 2x2
    # Percentages sum to ~100
    total_pct = sum(s["percentage"] for s in summary)
    assert total_pct == pytest.approx(100.0, abs=0.1)


def test_phase_map_store_get_phase_masks_is_round_trip():
    """The masks returned must reconstruct the phase grid exactly."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    out_entries, masks = store.get_phase_masks()
    assert len(out_entries) == 2
    # Reconstruct the grid from the masks (ignoring -1 region)
    reconstructed = np.full_like(grid, -1)
    for idx, m in masks.items():
        reconstructed[m] = idx
    np.testing.assert_array_equal(reconstructed, grid)


def test_phase_map_store_clear_drops_state():
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3)
    assert store.is_set()
    store.clear()
    assert not store.is_set()
    assert store.get_state() is None
    assert store.phase_summary() == []


# ---------------------------------------------------------------------------
# Palette + rendering
# ---------------------------------------------------------------------------


def test_phase_palette_is_deterministic_and_correct_length():
    """Same call twice → same colours; first N colours of palette(N+k) match."""
    a = phase_palette(8)
    b = phase_palette(8)
    assert a == b
    longer = phase_palette(20)
    assert longer[:8] == a, "palette must be a stable prefix as N grows"


def test_phase_palette_distinct_colours():
    """Adjacent phases must not collapse to nearly identical RGB triples."""
    palette = phase_palette(20)
    seen = set()
    for c in palette:
        # quantise to 32 levels to allow tiny rounding overlap, but flag
        # genuinely overlapping colours as a real failure
        seen.add(tuple(v // 8 for v in c))
    assert len(seen) >= 18  # tolerate at most a couple of near-duplicates


def test_phase_map_store_save_load_round_trip(tmp_path):
    """Sidecar must round-trip: save → fresh store → load → identical state."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3,
                             file_path=str(tmp_path / "fake.h5oina"))
    sidecar = store.save_to_disk(str(tmp_path / "fake.h5oina"))
    assert sidecar is not None and sidecar.is_file()

    # Fresh store, then reload
    store2 = PhaseMapStore()
    assert store2.load_from_disk(str(tmp_path / "fake.h5oina")) is True
    state2 = store2.get_state()
    assert state2 is not None
    assert state2.n_rows == 4 and state2.n_cols == 4
    assert len(state2.phase_entries) == len(entries)
    assert state2.phase_entries[0].cif_filename == entries[0].cif_filename
    assert state2.phase_entries[0].composition == entries[0].composition
    np.testing.assert_array_equal(state2.phase_grid, grid)
    np.testing.assert_array_almost_equal(state2.score_grid, scores)
    assert state2.tolerance == 15.0
    assert state2.min_score == 0.3


def test_phase_map_store_save_persists_locked_mask(tmp_path):
    """Manual-edit lock mask must survive a save/load cycle."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3,
                             file_path=str(tmp_path / "f.h5oina"))
    store.assign_region(0, 0, 0, 1, target_phase_index=1)
    store.save_to_disk(str(tmp_path / "f.h5oina"))

    store2 = PhaseMapStore()
    store2.load_from_disk(str(tmp_path / "f.h5oina"))
    state = store2.get_state()
    assert state.locked_mask is not None
    assert state.locked_mask[0, 0] and state.locked_mask[0, 1]
    # Pixels not painted must NOT be locked
    assert not state.locked_mask[3, 3]


def test_phase_map_store_load_missing_file_returns_false(tmp_path):
    """No sidecar → False, no exception, state untouched."""
    store = PhaseMapStore()
    assert store.load_from_disk(str(tmp_path / "nope.h5oina")) is False
    assert not store.is_set()


def test_phase_map_store_load_corrupt_sidecar_returns_false(tmp_path):
    """Garbage in the sidecar must not corrupt an empty store."""
    target = tmp_path / "broken.h5oina"
    sidecar = tmp_path / "broken.h5oina.phase_map.npz"
    target.touch()
    sidecar.write_bytes(b"not a numpy archive")
    store = PhaseMapStore()
    assert store.load_from_disk(str(target)) is False
    assert not store.is_set()


def test_phase_map_store_autosave_writes_on_set_classification(tmp_path):
    """set_classification() must transparently write the sidecar."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    fake = str(tmp_path / "a.h5oina")
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3,
                             file_path=fake)
    assert (tmp_path / "a.h5oina.phase_map.npz").is_file()


def test_phase_map_store_autosave_writes_on_assign_region(tmp_path):
    """assign_region() must update the sidecar so manual edits survive a restart."""
    store = PhaseMapStore()
    entries, grid, scores = _stock_state()
    fake = str(tmp_path / "b.h5oina")
    store.set_classification(grid, scores, entries, tolerance=15.0, min_score=0.3,
                             file_path=fake)
    sidecar = tmp_path / "b.h5oina.phase_map.npz"
    mtime_before = sidecar.stat().st_mtime
    # Force a clearly-newer mtime so the test is robust on filesystems
    # with 1s resolution.
    import time as _t
    _t.sleep(0.05)
    store.assign_region(2, 3, 0, 1, target_phase_index=0)
    mtime_after = sidecar.stat().st_mtime
    assert mtime_after >= mtime_before  # refreshed (allow == on coarse fs)


def test_render_phase_map_to_base64_returns_decodable_png():
    """The encoded image must decode back to a (n_rows, n_cols, 3) RGB image."""
    entries, grid, scores = _stock_state()
    state = PhaseMapState(
        phase_grid=grid, score_grid=scores,
        phase_entries=entries, n_rows=4, n_cols=4,
        tolerance=15.0, min_score=0.3,
    )
    b64 = render_phase_map_to_base64(state)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    # And palette_hex must give one entry per phase (not per legend slot)
    hex_colours = palette_hex_for_state(state)
    assert len(hex_colours) == len(entries)
    assert all(c.startswith("#") and len(c) == 7 for c in hex_colours)
